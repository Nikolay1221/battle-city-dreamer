"""
Monkey-patch for nes-py to fix NumPy 2.0 compatibility.

The issue: nes-py uses uint8 numpy scalars which overflow when multiplied
by 1024 in NumPy 2.0 (raises OverflowError instead of silent wrap).

Usage:
    import nes_py_patch  # Run this BEFORE importing nes_py or BattleCityEnv
"""

try:
    import nes_py._rom as rom_module

    def _patched_prg_rom_stop(self):
        """The exclusive stopping index of the PRG ROM (patched for NumPy 2.0)."""
        return int(self.prg_rom_start) + int(self.prg_rom_size) * 1024

    def _patched_chr_rom_stop(self):
        """The exclusive stopping index of the CHR ROM (patched for NumPy 2.0)."""
        return int(self.chr_rom_start) + int(self.chr_rom_size) * 1024

    # Apply patches
    rom_module.ROM.prg_rom_stop = property(_patched_prg_rom_stop)
    rom_module.ROM.chr_rom_stop = property(_patched_chr_rom_stop)

    print("✅ nes-py NumPy 2.0 compatibility patch applied!")
except ImportError:
    print("⚠️ nes-py not installed, skipping patch")
