use std::hint::black_box;

#[inline(always)]
pub fn stress_integer(iterations: u64, accumulator: &mut u64) {
    for i in 0..iterations {
        let x = black_box(i);
        let y = x.wrapping_mul(0x9e3779b97f4a7c15_u64);
        let z = y ^ (y >> 17);
        let w = z.rotate_left(31);
        *accumulator = black_box(accumulator.wrapping_add(w));
    }
}

#[inline(always)]
pub fn stress_float(iterations: u64, accumulator: &mut f64) {
    for i in 0..iterations {
        let x = black_box(i as f64 + 1.0);
        // Just keep the FPU busy, precision loss on large i is acceptable for stress
        let y = x.sqrt() * 1.618033988749895;
        let z = y.sin() + y.cos();
        let w = z.abs().ln_1p();
        *accumulator = black_box(*accumulator + w);
    }
}

/// Memory latency test - single pointer-chasing chain.
///
/// CRITICAL PERFORMANCE NOTE:
/// This function relies on the buffer length being a power of 2.
/// We use `& mask` instead of `% len` to avoid the expensive `DIV` instruction.
/// A `DIV` instruction can take 20-50 cycles, which would pollute the latency
/// measurement (typically 100-300 cycles for DRAM). The bitwise AND is < 1
/// cycle.
#[inline(always)]
pub fn stress_memory_latency(iterations: u64, buffer: &mut [u64]) {
    let len = buffer.len();
    if len == 0 {
        return;
    }

    let mask = len - 1;
    let mut index = 0usize;

    for i in 0..iterations {
        // Read value from DRAM/Cache
        let value = black_box(buffer[index]);

        // Calculate next value (ALU op)
        let new_value = value.wrapping_mul(6364136223846793005_u64).wrapping_add(i);

        // Write back
        buffer[index] = black_box(new_value);

        // Calculate next index dependent on the read value (Pointer chasing)
        // The XOR prevents the prefetcher from guessing the stride.
        // The AND mask forces it within bounds without a DIV instruction.
        index = black_box(((new_value >> 17) ^ i) as usize & mask);
    }
}

/// Memory bandwidth test - parallel independent streams
#[inline(always)]
pub fn stress_memory_bandwidth(iterations: u64, buffer: &mut [u64]) {
    let len = buffer.len();
    if len == 0 {
        return;
    }

    let mask = len - 1;

    // Modern memory controllers can handle 8-16 parallel requests
    const STREAMS: usize = 8;
    let mut indices = [0usize; STREAMS];

    // Different Linear Congruential Generators (LCG) multipliers for each stream
    const LCG_MULTS: [u64; STREAMS] = [
        6364136223846793005,
        2862933555777941757,
        3202034522624059733,
        7046029254386353087,
        5495735621104509439,
        1865811235122147685,
        8121734705789632447,
        4976774832059184573,
    ];

    // Initialize streams at different buffer offsets
    for (i, idx) in indices.iter_mut().enumerate() {
        *idx = (len / STREAMS) * i;
    }

    for iter in 0..iterations {
        let mut values = [0u64; STREAMS];

        // 1. Burst Read
        for stream_id in 0..STREAMS {
            values[stream_id] = black_box(buffer[indices[stream_id]]);
        }

        // 2. ALU Ops
        let mut new_values = [0u64; STREAMS];
        for stream_id in 0..STREAMS {
            new_values[stream_id] = values[stream_id]
                .wrapping_mul(LCG_MULTS[stream_id])
                .wrapping_add(iter);
        }

        // 3. Burst Write
        for stream_id in 0..STREAMS {
            buffer[indices[stream_id]] = black_box(new_values[stream_id]);
        }

        // 4. Update Indices (Pointer chasing per stream)
        for stream_id in 0..STREAMS {
            indices[stream_id] = black_box(((new_values[stream_id] >> 17) as usize) & mask);
        }
    }
}

/// Allocates a memory buffer that is strictly sized to a power of 2.
///
/// We round DOWN to the nearest power of 2 relative to the requested MB size.
/// This ensures we don't exceed the safety caps calculated in `system.rs`,
/// while ensuring that bitwise masking (& mask) works correctly in the
/// stress tests to avoid DIV instructions.
pub fn allocate_memory_buffer(size_mb: usize) -> Box<[u64]> {
    // Convert MB to bytes
    let requested_bytes = size_mb
        .checked_mul(1024 * 1024)
        .expect("Requested memory size too large");

    let elem_size = std::mem::size_of::<u64>();
    let requested_elements = requested_bytes / elem_size;

    // Round down to nearest power of 2 elements to ensure safe masking
    // If requested_elements is 0, we default to a small buffer (e.g., 4096 items)
    let num_elements = if requested_elements == 0 {
        4096
    } else {
        // prev_power_of_two isn't stable on usize in all versions, so we use next >> 1
        // logic
        let p2 = requested_elements.next_power_of_two();
        if p2 > requested_elements { p2 / 2 } else { p2 }
    };

    // Ensure a minimum practical size (e.g., 1MB = 131k u64s) to avoid trivial
    // loops
    let num_elements = num_elements.max(128 * 1024);

    let mut buffer = Vec::with_capacity(num_elements);
    // Fill to capacity to actually allocate the pages
    for i in 0..num_elements {
        buffer.push((i as u64) ^ 0xdeadbeef);
    }
    buffer.into_boxed_slice()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocation_is_power_of_two() {
        // 100 MB -> ~12.5M u64s. Nearest lower power of 2 is 8M (8388608).
        let buffer = allocate_memory_buffer(100);
        assert!(buffer.len().is_power_of_two());
        assert!(buffer.len() > 0);
    }

    #[test]
    fn test_stress_integer_prevents_optimization() {
        let mut acc = 0u64;
        stress_integer(1000, &mut acc);
        assert_ne!(acc, 0);
    }

    #[test]
    fn test_stress_memory_latency_modifies_buffer() {
        let mut buffer = allocate_memory_buffer(1); // 1MB
        stress_memory_latency(10000, &mut buffer);
        let non_zero_count = buffer.iter().filter(|&&x| x != 0).count();
        assert!(non_zero_count > 0);
    }

    #[test]
    fn test_stress_memory_bandwidth_modifies_buffer() {
        let mut buffer = allocate_memory_buffer(1); // 1MB
        stress_memory_bandwidth(5000, &mut buffer);
        let non_zero_count = buffer.iter().filter(|&&x| x != 0).count();
        assert!(non_zero_count > 0);
    }
}
