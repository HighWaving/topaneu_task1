import nibabel as nib


def reorient_nii(orig_nii, targ_aff="LPS"):
    """
    reorientate to the standard LPS+ DICOM coord
    see links:
    https://github.com/nipy/nibabel/issues/1010
    https://docs.monai.io/en/0.1.0/_modules/monai/data/nifti_writer.html
    """
    # if affine already in targ_aff, do not reorient
    if "".join(nib.aff2axcodes(orig_nii.affine)) == targ_aff:
        print(f"already in {targ_aff}, no need to reorient")
        return orig_nii

    print("before reorient_nii")
    _log_ornt(orig_nii)

    # You need to find the transform from
    # where you are io_orientation(affine)
    # to where you want to be axcodes2ornt("LPS")
    orig_ornt = nib.io_orientation(orig_nii.affine)
    targ_ornt = nib.orientations.axcodes2ornt(targ_aff)
    transform = nib.orientations.ornt_transform(orig_ornt, targ_ornt)

    # NOTE: ornt_transform returns orientation that
    # transforms from start_ornt to end_ornt
    # orientation in nibabel is a (p, 2) ndarray
    # see https://nipy.org/nibabel/reference/nibabel.orientations.html

    img_orient = orig_nii.as_reoriented(transform)

    print("after reorient_nii")
    _log_ornt(img_orient)

    return img_orient


def _log_ornt(loaded_nii):
    print(
        "loaded_nii orientation:",
        nib.orientations.ornt2axcodes(
            nib.orientations.io_orientation(loaded_nii.affine)
        ),
    )
    print("loaded_nii aff2axcodes:", nib.aff2axcodes(loaded_nii.affine))


if __name__ == "__main__":
    import argparse
    import re
    import sys
    from pathlib import Path

    print("\n##### reorient_nii.py #####")

    # first element of sys.argv represents the script name
    parser = argparse.ArgumentParser(
        usage="%(prog)s [nii_file] [orint_code]",
        description="reorient ROI according to orientation code",
    )
    parser.add_argument(
        "nii_file",
    )
    parser.add_argument(
        "orint_code",
    )
    args = parser.parse_args(sys.argv[1:])
    nii_file = args.nii_file
    orint_code = args.orint_code

    print("args parse nii_file = ", nii_file)
    print("args parse orint_code (default LPS) = ", orint_code)

    orig_nii = nib.load(nii_file)
    img_orient = reorient_nii(orig_nii, targ_aff=orint_code)

    pattern = r"\.nii"
    stem = re.sub(pattern, "", Path(nii_file).stem) + f"_{orint_code}"

    filename = f"{stem}.nii.gz"
    img_orient.to_filename(filename)

    # DONE!
    print(f"{filename} SAVED!\n")
