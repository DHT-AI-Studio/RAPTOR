/* memvid-sdk's aarch64 wheel calls GCC's outline-atomics helper
 * __aarch64_cas8_sync but never links it in (confirmed: not defined
 * anywhere in the runtime image -- not libgcc_s, not libatomic, not
 * python3.11 itself, not any installed site-package .so, and every
 * memvid-sdk release from 2.0.152 to 2.0.160 has the same gap). Standard
 * ABI (see gcc/libgcc/config/aarch64/lse.S): atomic compare-and-swap,
 * always returns the value that was actually at *ptr before the call.
 */
#include <stdatomic.h>
#include <stdint.h>

uint64_t __aarch64_cas8_sync(uint64_t old, uint64_t new_val, volatile void *ptr) {
    atomic_compare_exchange_strong_explicit(
        (volatile _Atomic(uint64_t) *)ptr, &old, new_val,
        memory_order_seq_cst, memory_order_seq_cst);
    return old;
}
