use std::hint::black_box;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::workload::{
    allocate_memory_buffer,
    stress_float,
    stress_integer,
    stress_memory_bandwidth,
    stress_memory_latency,
};

pub fn worker_thread(
    id: usize,
    stop_flag: Arc<AtomicBool>,
    work_counter: Arc<AtomicU64>,
    workload: &str,
    batch_size: u64,
    memory_mb: usize,
) {
    let mut int_acc = id as u64;
    let mut float_acc = id as f64;

    // This allocates a buffer sized to a power of 2 to allow efficient masking
    let mut mem_buffer = allocate_memory_buffer(memory_mb);

    loop {
        // On x86 (strong memory model), Relaxed is often sufficient for a boolean flag,
        // but on ARM/Apple Silicon (weak memory model), Relaxed loads might not observe
        // the store from the main thread immediately, leading to delays in stopping.
        // Acquire guarantees that we see all Release stores that happened before.
        if stop_flag.load(Ordering::Acquire) {
            break;
        }

        match workload {
            "integer" => stress_integer(batch_size, &mut int_acc),
            "float" => stress_float(batch_size, &mut float_acc),
            // "memory" maps to latency test by default as it's more intensive on the controller
            // logic
            "memory" | "memory-latency" => stress_memory_latency(batch_size, &mut mem_buffer),
            "memory-bandwidth" => stress_memory_bandwidth(batch_size, &mut mem_buffer),
            _ => {
                // Mixed workload: 33% split
                stress_integer(batch_size / 3, &mut int_acc);
                stress_float(batch_size / 3, &mut float_acc);
                stress_memory_latency(batch_size / 3, &mut mem_buffer);
            },
        }

        work_counter.fetch_add(batch_size, Ordering::Relaxed);
    }

    // Prevent dead code elimination of the accumulators and buffer
    black_box(int_acc);
    black_box(float_acc);
    black_box(mem_buffer);
}

#[cfg(test)]
mod tests {
    use std::thread;
    use std::time::Duration;

    use super::*;

    #[test]
    fn test_worker_respects_stop_flag() {
        let stop = Arc::new(AtomicBool::new(false));
        let counter = Arc::new(AtomicU64::new(0));

        let stop_clone = Arc::clone(&stop);
        let counter_clone = Arc::clone(&counter);

        let handle = thread::spawn(move || {
            worker_thread(0, stop_clone, counter_clone, "integer", 10000, 1);
        });

        thread::sleep(Duration::from_millis(50));
        stop.store(true, Ordering::Release);

        handle.join().expect("Worker should terminate cleanly");
        assert!(counter.load(Ordering::Relaxed) > 0);
    }
}
