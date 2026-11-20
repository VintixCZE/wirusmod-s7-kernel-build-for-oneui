#!/usr/bin/env python3
"""Validate a Galaxy S7 (Exynos 8890) boot image before it is published.

Usage:
  validate_boot.py --zip FLASHABLE.zip --stock boot.zip [--report FILE]
  validate_boot.py --img boot.img      --stock boot.zip [--report FILE]

Hard failures (exit 1) are things that make S-Boot reject the image or that
make a flash silently truncate it:
  * image larger than the BOOT partition (40 MiB on hero/hero2 devices)
  * kernel larger than 32 MiB (suspected S-Boot limit, see KERNEL_LIMIT)
  * header / payload layout that does not fit inside the file
  * kernel is not a raw arm64 Image, ramdisk is not a valid gzip cpio,
    DTB image is not a dtbTool ("DTBH") container
  * SEANDROIDENFORCE footer missing
  * boot addresses / page size differ from the known-good stock image

Anything that is only a comparison against the stock image (kernel bigger than
the stock kernel) is reported as a WARNING with exact numbers, because the
stock kernel size is a known-good reference, not a documented limit.
"""
import argparse
import gzip
import io
import os
import struct
import sys
import tarfile
import zipfile

# BOARD_BOOTIMAGE_PARTITION_SIZE of LineageOS device/samsung/hero-common
# (board/fs.mk): 41943040 = 40 MiB.  S7 / S7 Edge share this layout.
BOOT_PARTITION_SIZE = 41943040
# Suspected S-Boot limit on the kernel payload.  Evidence, not proof: the only
# known-good kernel (stock XFire, 33,374,744 bytes) sits 179,688 bytes below
# 32 MiB, while the CI build that failed with "Invalid kernel length" had a
# 35,605,288 byte kernel (its whole image still fit the 40 MiB partition).
KERNEL_LIMIT = 32 * 1024 * 1024
PAGE = 2048
FOOTER = b"SEANDROIDENFORCE"


def find_boot(blob, depth=0):
    """Return the first Android boot image found in blob (recursing into zips)."""
    if blob.startswith(b"ANDROID!"):
        return blob
    if depth >= 3:
        return None
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            for item in z.infolist():
                if item.is_dir():
                    continue
                found = find_boot(z.read(item), depth + 1)
                if found is not None:
                    return found
    except (zipfile.BadZipFile, OSError):
        pass
    return None


def boot_from_flashable(path):
    with zipfile.ZipFile(path) as z:
        archive = z.read("moro/kernel.tar.xz")
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:xz") as t:
        boots = [m for m in t.getmembers() if m.isfile() and m.name.endswith("-boot.img")]
        if len(boots) != 1:
            raise SystemExit("FAIL: expected exactly one *-boot.img in kernel.tar.xz, found %d" % len(boots))
        name = boots[0].name
        return name, t.extractfile(boots[0]).read()


def align(n, page=PAGE):
    return (n + page - 1) // page * page


KEYS = ("kernel_size", "kernel_addr", "ramdisk_size", "ramdisk_addr", "second_size",
        "second_addr", "tags_addr", "page_size", "dt_size", "os_version")


def parse(data, label, problems):
    if len(data) < 2048 or data[:8] != b"ANDROID!":
        problems.append("%s: not an Android boot image" % label)
        return None
    h = dict(zip(KEYS, struct.unpack_from("<10I", data, 8)))
    ps = h["page_size"]
    h["kernel_off"] = ps
    h["ramdisk_off"] = h["kernel_off"] + align(h["kernel_size"], ps)
    h["second_off"] = h["ramdisk_off"] + align(h["ramdisk_size"], ps)
    h["dt_off"] = h["second_off"] + align(h["second_size"], ps)
    h["payload_end"] = h["dt_off"] + align(h["dt_size"], ps)
    h["total"] = len(data)
    return h


def check_image(data, label, problems, warnings):
    h = parse(data, label, problems)
    if h is None:
        return None
    if h["page_size"] != PAGE:
        problems.append("%s: page size %d, expected %d" % (label, h["page_size"], PAGE))
        return h
    if h["kernel_size"] == 0:
        problems.append("%s: kernel_size is 0" % label)
    if h["payload_end"] > h["total"]:
        problems.append("%s: header says payload ends at %d but the file is only %d bytes "
                        "(kernel/ramdisk/dt sizes do not fit the file -> 'Invalid kernel length')"
                        % (label, h["payload_end"], h["total"]))
        return h

    kernel = data[h["kernel_off"]:h["kernel_off"] + h["kernel_size"]]
    if kernel[:2] == b"\x1f\x8b":
        problems.append("%s: kernel is gzip-compressed; this device boots a raw arm64 Image" % label)
    elif kernel[56:60] != b"ARM\x64":
        problems.append("%s: kernel has no arm64 Image magic at offset 56" % label)
    else:
        text_offset, image_size = struct.unpack_from("<QQ", kernel, 8)
        h["image_text_offset"] = text_offset
        h["image_size"] = image_size
        if image_size < h["kernel_size"]:
            problems.append("%s: Image header image_size (%d) is smaller than kernel_size (%d)"
                            % (label, image_size, h["kernel_size"]))

    ramdisk = data[h["ramdisk_off"]:h["ramdisk_off"] + h["ramdisk_size"]]
    try:
        cpio = gzip.decompress(ramdisk)
        if not cpio.startswith(b"070701"):
            problems.append("%s: ramdisk decompresses but is not a newc cpio" % label)
        h["ramdisk_unpacked"] = len(cpio)
    except Exception as exc:  # noqa: BLE001 - any gzip error is a hard failure
        problems.append("%s: ramdisk is not a valid gzip stream (%s)" % (label, exc))

    if h["dt_size"] == 0 or data[h["dt_off"]:h["dt_off"] + 4] != b"DTBH":
        problems.append("%s: dt image is missing or is not a dtbTool 'DTBH' container" % label)

    tail = data[h["payload_end"]:]
    h["tail"] = tail
    if FOOTER not in tail:
        problems.append("%s: SEANDROIDENFORCE footer missing (Samsung bootloader will refuse it)" % label)
    return h


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--zip", help="flashable ZIP produced by build.sh")
    src.add_argument("--img", help="boot.img to check directly")
    ap.add_argument("--stock", required=True, help="stock reference boot.zip (or boot.img)")
    ap.add_argument("--report", help="write the text report to this file")
    ap.add_argument("--kernel-limit-mib", type=float, default=KERNEL_LIMIT / 1048576,
                    help="fail if the kernel is larger than this many MiB (default 32, 0 = report only)")
    ap.add_argument("--summary", default=os.environ.get("GITHUB_STEP_SUMMARY"),
                    help="markdown file to append a table to (defaults to $GITHUB_STEP_SUMMARY)")
    args = ap.parse_args()

    problems, warnings = [], []
    kernel_limit = int(args.kernel_limit_mib * 1048576)

    if args.zip:
        name, image = boot_from_flashable(args.zip)
    else:
        name = os.path.basename(args.img)
        with open(args.img, "rb") as f:
            image = f.read()
    with open(args.stock, "rb") as f:
        stock = find_boot(f.read())
    if stock is None:
        raise SystemExit("FAIL: could not find a boot.img inside the stock reference " + args.stock)

    stock_problems = []
    s = check_image(stock, "stock", stock_problems, [])
    if stock_problems:
        raise SystemExit("FAIL: stock reference itself looks broken: " + "; ".join(stock_problems))
    b = check_image(image, "built", problems, warnings)

    if b is not None:
        for key in ("kernel_addr", "ramdisk_addr", "second_addr", "tags_addr", "page_size"):
            if b[key] != s[key]:
                problems.append("built header %s = %#x, stock = %#x" % (key, b[key], s[key]))

        if b["total"] > BOOT_PARTITION_SIZE:
            problems.append(
                "boot image is %d bytes, %d bytes larger than the %d byte (40 MiB) BOOT partition. "
                "dd/Odin would truncate it and S-Boot reports 'Invalid kernel length'."
                % (b["total"], b["total"] - BOOT_PARTITION_SIZE, BOOT_PARTITION_SIZE))
        if kernel_limit and b["kernel_size"] > kernel_limit:
            problems.append(
                "kernel is %d bytes, %d bytes above the %d byte limit. The known-good stock kernel is %d bytes "
                "(%d below 32 MiB); kernels above 32 MiB are suspected to be rejected by S-Boot with "
                "'Invalid kernel length'. Shrink the kernel (see ci/trim-*.config)."
                % (b["kernel_size"], b["kernel_size"] - kernel_limit, kernel_limit, s["kernel_size"],
                   KERNEL_LIMIT - s["kernel_size"]))
        elif b["kernel_size"] > s["kernel_size"]:
            warnings.append(
                "kernel is %d bytes, %d bytes larger than the stock reference kernel (%d). The stock kernel "
                "is a known-good size, not a proven limit; if the device rejects the image, shrink the kernel."
                % (b["kernel_size"], b["kernel_size"] - s["kernel_size"], s["kernel_size"]))
        if "tail" in b and b["tail"] != s.get("tail"):
            def preview(t):
                return "%d bytes, starts %r" % (len(t), t[:40])
            warnings.append("trailing bytes after the DT image differ from stock (built: %s; stock: %s)"
                            % (preview(b["tail"]), preview(s.get("tail", b""))))
        if b.get("image_text_offset") not in (None, s.get("image_text_offset")):
            warnings.append("Image text_offset %#x differs from stock %#x"
                            % (b["image_text_offset"], s.get("image_text_offset", 0)))

    lines = ["Boot image validation: %s" % name, "=" * 60]
    if b is not None and s is not None:
        rows = [
            ("boot.img total", b["total"], s["total"], BOOT_PARTITION_SIZE),
            ("kernel_size", b["kernel_size"], s["kernel_size"], kernel_limit or None),
            ("Image image_size (incl. bss)", b.get("image_size"), s.get("image_size"), None),
            ("ramdisk_size", b["ramdisk_size"], s["ramdisk_size"], None),
            ("dt_size", b["dt_size"], s["dt_size"], None),
        ]
        lines.append("%-30s %14s %14s %16s" % ("", "built", "stock", "limit"))
        for label, bv, sv, limit in rows:
            lines.append("%-30s %14s %14s %16s" % (label, bv if bv is not None else "-",
                                                    sv if sv is not None else "-",
                                                    limit if limit else ""))
        if b["total"] <= BOOT_PARTITION_SIZE:
            lines.append("Headroom in BOOT partition: %d bytes" % (BOOT_PARTITION_SIZE - b["total"]))
    for w in warnings:
        lines.append("WARNING: " + w)
    for p in problems:
        lines.append("FAIL: " + p)
    lines.append("RESULT: " + ("FAIL" if problems else "PASS"))
    report = "\n".join(lines)
    print(report)

    if args.report:
        with open(args.report, "w") as f:
            f.write(report + "\n")
    if args.summary:
        with open(args.summary, "a") as f:
            f.write("### Boot image validation: %s\n\n```\n%s\n```\n" % ("FAIL" if problems else "PASS", report))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
