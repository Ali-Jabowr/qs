# For Ali — preparing the SSD before the friend leaves

The friend's kit is the Russian files `README.md` … `07-sdat.md`. This page is for us: what must be
true about the SSD before it is handed over. Done together in a Claude session once the SATA→USB
adapter arrives.

## The drive

DEXP C100SMYM1024 (1 TB, 2.5" SATA, 550/495 MB/s) behind a **USB 3.0 + UASP** adapter.

## Install (in a VM that sees only the SSD)

- [ ] **Ubuntu 24.04 LTS**, installed from the ISO inside a QEMU VM with the SSD passed through as
  the only disk. This way the installer can't write a bootloader onto the laptop's internal disks
- [ ] Bootloader also at the removable path `EFI/BOOT/BOOTX64.EFI`, so it boots on any UEFI PC
  (the friend's laptop has never seen it)
- [ ] Full-disk encryption (LUKS). The password goes on paper, kept separately from the SSD
- [ ] User `qs`, password on the same paper
- [ ] Packages: `v4l-utils rsync python3-tk git` (+ `uv`)
- [ ] Cloud sync and auto-updates off, so nothing starts downloading or uploading on the factory network

## The project on it

- [ ] Repo cloned to `/home/qs/QS` at the same commit as here, with `kit-moscow/` included
- [ ] `uv sync` → `.venv` present; `bin/qs --help` runs
- [ ] `assets/DejaVuSans.ttf` present (the shoot screen needs it)
- [ ] No ML weights needed on site: the shoot doesn't run PatchCore
- [ ] `data/raw/` is empty: no bench images go to Moscow

## Test on the friend's laptop, at home (the rehearsal in `01-sobrat.md`)

- [ ] Boots from the SSD (note the boot-menu key in `02-noutbuk.md`)
- [ ] `./kit-moscow/kamera.sh` → PASS controls / frames / read-back
- [ ] `./kit-moscow/foto.sh repeticiya` → a frame saved, readable
- [ ] `./kit-moscow/semka.sh repeticiya-plan` → plan made (then delete `data/raw/repeticiya-plan`)
- [ ] `./kit-moscow/kopiya.sh` → "ГОТОВО"
- [ ] A Tk window opens on the friend's screen (Wayland via XWayland is fine)

## The backup stick

The Kingston 115 GB stick (now holding the Omarchy ISO) can be the backup drive, **once you've
decided about the old backups**. Format it as exFAT, label `QS-BACKUP`.

## Known from testing on 11.10

- Right after the camera opens, the C920 can read back gain 24 for a moment when 25 was set.
  `snap.py` waits 3 s and retries 3×. `qs shoot` and `camera-test` passed on the same camera.
- `qs shoot` cannot resume a started session (#33). The fallback is the `foto.sh dosemka` route
  in `06-esli-problema.md`, with frames named like the plan; they get imported by hand afterwards.
