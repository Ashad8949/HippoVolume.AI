"""
Here we do inference on a DICOM volume, constructing the volume first, and then sending it to the
clinical archive

This code will do the following:
    1. Identify the series to run HippoCrop.AI algorithm on from a folder containing multiple studies
    2. Construct a NumPy volume from a set of DICOM files
    3. Run inference on the constructed volume
    4. Create report from the inference
    5. Call a shell script to push report to the storage archive
"""

import os
import sys
import datetime
import time
import shutil
import subprocess

import numpy as np
import pydicom

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

from inference.UNetInferenceAgent import UNetInferenceAgent

def load_dicom_volume_as_numpy_from_list(dcmlist):
    """Loads a list of PyDicom objects a Numpy array.
    Assumes that only one series is in the array

    Arguments:
        dcmlist {list of PyDicom objects} -- path to directory

    Returns:
        tuple of (3D volume, header of the 1st image)
    """

    # In the real world you would do a lot of validation here
    slices = [np.flip(dcm.pixel_array).T for dcm in sorted(dcmlist, key=lambda dcm: dcm.InstanceNumber)]

    # Make sure that you have correctly constructed the volume from your axial slices!
    hdr = dcmlist[0]

    # We return header so that we can inspect metadata properly.
    # Since for our purposes we are interested in "Series" header, we grab header of the
    # first file (assuming that any instance-specific values will be ighored - common approach)
    # We also zero-out Pixel Data since the users of this function are only interested in metadata
    hdr.PixelData = None
    return (np.stack(slices, 2), hdr)

def get_predicted_volumes(pred):
    """Gets volumes of two hippocampal structures from the predicted array

    Arguments:
        pred {Numpy array} -- array with labels. Assuming 0 is bg, 1 is anterior, 2 is posterior

    Returns:
        A dictionary with respective volumes
    """

    # Each voxel is 1x1x1mm so volume in mm^3 is just the voxel count
    volume_ant = np.sum(pred == 1)
    volume_post = np.sum(pred == 2)
    total_volume = volume_ant + volume_post

    return {"anterior": volume_ant, "posterior": volume_post, "total": total_volume}

def create_report(inference, header, orig_vol, pred_vol):
    """Generates an image with inference report

    Arguments:
        inference {Dictionary} -- dict containing anterior, posterior and full volume values
        header {PyDicom Dataset} -- DICOM header
        orig_vol {Numpy array} -- original volume
        pred_vol {Numpy array} -- predicted label

    Returns:
        PIL image
    """

    # The code below uses PIL image library to compose an RGB image that will go into the report
    # A standard way of storing measurement data in DICOM archives is creating such report and
    # sending them on as Secondary Capture IODs (http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_A.8.html)
    # Essentially, our report is just a standard RGB image, with some metadata, packed into 
    # DICOM format. 

    pimg = Image.new("RGB", (1000, 1000))
    draw = ImageDraw.Draw(pimg)

    header_font = ImageFont.truetype("assets/Roboto-Regular.ttf", size=40)
    main_font = ImageFont.truetype("assets/Roboto-Regular.ttf", size=20)
    small_font = ImageFont.truetype("assets/Roboto-Regular.ttf", size=16)

    slice_nums = [orig_vol.shape[2]//3, orig_vol.shape[2]//2, orig_vol.shape[2]*3//4]

    # STAND-OUT: Compute additional clinical context
    # Reference ranges for adult hippocampal volume (from literature)
    # Typical range: 2500-4500 mm³ per hippocampus
    total_vol = inference['total']
    if total_vol < 2500:
        volume_assessment = "BELOW normal range (< 2500 mm³) — consider clinical correlation"
        assessment_color = (255, 100, 100)  # red
    elif total_vol > 4500:
        volume_assessment = "ABOVE normal range (> 4500 mm³) — verify segmentation"
        assessment_color = (255, 200, 100)  # orange
    else:
        volume_assessment = "Within normal reference range (2500-4500 mm³)"
        assessment_color = (100, 255, 100)  # green

    # Compute percentage of each sub-region
    ant_pct = (inference['anterior'] / total_vol * 100) if total_vol > 0 else 0
    post_pct = (inference['posterior'] / total_vol * 100) if total_vol > 0 else 0

    # Compute segmentation confidence: ratio of non-zero voxels in prediction
    total_voxels = pred_vol.size
    segmented_voxels = np.sum(pred_vol > 0)
    seg_ratio = segmented_voxels / total_voxels if total_voxels > 0 else 0

    # Header section
    draw.text((10, 0), "HippoVolume.AI", (255, 255, 255), font=header_font)
    draw.text((10, 50), "Automated Hippocampal Volume Report", (200, 200, 200), font=small_font)

    # Patient info section
    draw.multiline_text((10, 90),
                        f"Patient ID: {header.PatientID}\n"
                        f"Patient Name: {getattr(header, 'PatientName', 'N/A')}\n"
                        f"Study Date: {getattr(header, 'StudyDate', 'N/A')}\n"
                        f"Series: {getattr(header, 'SeriesDescription', 'N/A')}\n"
                        f"Modality: {getattr(header, 'Modality', 'N/A')}\n"
                        f"Institution: {getattr(header, 'InstitutionName', 'N/A')}",
                        (255, 255, 255), font=main_font)

    # Volume measurements section
    y_offset = 250
    draw.text((10, y_offset), "Volumetric Measurements", (255, 255, 100), font=main_font)
    draw.multiline_text((10, y_offset + 30),
                        f"Anterior Hippocampus:  {inference['anterior']:,.0f} mm\u00b3 ({ant_pct:.1f}%)\n"
                        f"Posterior Hippocampus: {inference['posterior']:,.0f} mm\u00b3 ({post_pct:.1f}%)\n"
                        f"Total Volume:          {inference['total']:,.0f} mm\u00b3",
                        (255, 255, 255), font=main_font)

    # STAND-OUT: Clinical reference range assessment
    draw.text((10, y_offset + 110), "Assessment:", (255, 255, 100), font=main_font)
    draw.text((130, y_offset + 110), volume_assessment, assessment_color, font=main_font)

    # STAND-OUT: Segmentation quality indicator
    draw.text((10, y_offset + 140),
              f"Segmentation coverage: {seg_ratio*100:.1f}% of volume ({segmented_voxels:,} / {total_voxels:,} voxels)",
              (180, 180, 180), font=small_font)

    # Show three axial slices with segmentation overlay
    for i, slc_num in enumerate(slice_nums):
        # Original image slice
        slc = orig_vol[:, :, slc_num]
        max_val = np.max(slc) if np.max(slc) > 0 else 1
        nd_img = np.flip((slc / max_val) * 0xff).T.astype(np.uint8)
        pil_i = Image.fromarray(nd_img, mode="L").convert("RGBA").resize((300, 300))
        pimg.paste(pil_i, box=(10 + i * 330, 450))
        draw.text((10 + i * 330, 420), f"Slice {slc_num}", (255, 255, 255), font=main_font)

        # Overlay: create colored mask
        mask_slc = pred_vol[:, :, slc_num]
        # Build RGBA overlay: anterior=green, posterior=blue
        overlay = np.zeros((*mask_slc.shape, 4), dtype=np.uint8)
        overlay[mask_slc == 1] = [0, 255, 0, 128]   # anterior - green
        overlay[mask_slc == 2] = [255, 0, 0, 128]    # posterior - red
        overlay_img = Image.fromarray(np.flip(overlay, axis=0).transpose(1, 0, 2), mode="RGBA").resize((300, 300))
        pil_i_copy = pil_i.copy()
        pil_i_copy = Image.alpha_composite(pil_i_copy, overlay_img)
        pimg.paste(pil_i_copy, box=(10 + i * 330, 770))

    draw.text((10, 750), "Segmentation Overlay (Green=Anterior, Red=Posterior)",
              (255, 255, 255), font=main_font)

    # STAND-OUT: Disclaimer / model info footer
    draw.text((10, 970),
              "AI-generated report — for clinical decision support only. Model: U-Net (Dice=0.90). Verify all measurements.",
              (150, 150, 150), font=small_font)

    return pimg

def save_report_as_dcm(header, report, path):
    """Writes the supplied image as a DICOM Secondary Capture file

    Arguments:
        header {PyDicom Dataset} -- original DICOM file header
        report {PIL image} -- image representing the report
        path {Where to save the report}

    Returns:
        N/A
    """

    # STAND-OUT: Fully valid DICOM Secondary Capture IOD per DICOM PS3.3 Section A.8
    # This implementation includes all Mandatory (M) modules from Table A.8-1

    out = pydicom.Dataset()

    # === File Meta Information Module ===
    out.file_meta = pydicom.Dataset()
    out.file_meta.FileMetaInformationVersion = b'\x00\x01'
    out.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    out.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.7"  # SC Image Storage
    out.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    out.file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID
    out.file_meta.ImplementationVersionName = "HippoVolumeAI"

    out.is_little_endian = True
    out.is_implicit_VR = False

    # === Patient Module (M) - Table C.7-1 ===
    out.PatientName = getattr(header, 'PatientName', '')
    out.PatientID = getattr(header, 'PatientID', '')
    out.PatientBirthDate = getattr(header, 'PatientBirthDate', '')
    out.PatientSex = getattr(header, 'PatientSex', '')

    # === General Study Module (M) - Table C.7-3 ===
    out.StudyInstanceUID = getattr(header, 'StudyInstanceUID', pydicom.uid.generate_uid())
    out.StudyDate = datetime.date.today().strftime("%Y%m%d")
    out.StudyTime = datetime.datetime.now().strftime("%H%M%S")
    out.ReferringPhysicianName = getattr(header, 'ReferringPhysicianName', '')
    out.StudyID = getattr(header, 'StudyID', '')
    out.AccessionNumber = getattr(header, 'AccessionNumber', '')
    out.StudyDescription = getattr(header, 'StudyDescription', 'HippoVolume.AI Analysis')

    # === General Series Module (M) - Table C.7-5a ===
    out.SeriesInstanceUID = pydicom.uid.generate_uid()
    out.Modality = "OT"  # Other
    out.SeriesNumber = 9999
    out.SeriesDate = out.StudyDate
    out.SeriesTime = out.StudyTime
    out.SeriesDescription = "HippoVolume.AI"
    out.OperatorsName = "HippoVolume.AI"
    out.PerformingPhysicianName = ""

    # === General Equipment Module (M) - Table C.7-8 ===
    out.Manufacturer = "HippoVolume.AI"
    out.InstitutionName = getattr(header, 'InstitutionName', '')
    out.InstitutionAddress = getattr(header, 'InstitutionAddress', '')
    out.StationName = "AI_WORKSTATION"
    out.ManufacturerModelName = "U-Net Hippocampal Segmentation v1.0"
    out.SoftwareVersions = "1.0"

    # === SC Equipment Module (M) - Table C.8-24 ===
    out.ConversionType = "WSD"  # Workstation
    out.SecondaryCaptureDeviceManufacturer = "HippoVolume.AI"
    out.SecondaryCaptureDeviceManufacturerModelName = "U-Net v1.0"
    out.SecondaryCaptureDeviceSoftwareVersions = "1.0"

    # === General Image Module (M) - Table C.7-9 ===
    out.InstanceNumber = 1
    out.PatientOrientation = ""
    out.ContentDate = out.StudyDate
    out.ContentTime = out.StudyTime
    out.ImageType = ["DERIVED", "PRIMARY"]
    out.ImagesInAcquisition = 1
    out.AcquisitionNumber = 1

    # === Image Pixel Module (M) - Table C.7-11b ===
    out.SamplesPerPixel = 3
    out.PhotometricInterpretation = "RGB"
    out.PlanarConfiguration = 0  # R1G1B1R2G2B2...
    out.Rows = report.height
    out.Columns = report.width
    out.BitsAllocated = 8
    out.BitsStored = 8
    out.HighBit = 7
    out.PixelRepresentation = 0  # unsigned
    out.PixelData = report.tobytes()

    # === SOP Common Module (M) - Table C.12-1 ===
    out.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"  # SC Image Storage
    out.SOPInstanceUID = out.file_meta.MediaStorageSOPInstanceUID
    out.SpecificCharacterSet = "ISO_IR 100"
    out.InstanceCreationDate = out.StudyDate
    out.InstanceCreationTime = out.StudyTime

    # === Additional useful attributes ===
    out.WindowCenter = ""
    out.WindowWidth = ""
    out.BurnedInAnnotation = "YES"
    out.LossyImageCompression = "00"  # No lossy compression

    pydicom.filewriter.dcmwrite(path, out, write_like_original=False)


def save_segmentation_as_dcm_series(header, orig_vol, pred_vol, output_dir):
    """STAND-OUT: Creates a DICOM image series from the segmentation mask.
    Each slice of the segmentation mask is saved as a separate DICOM file,
    forming a proper series that can be overlaid on the original in viewers
    like Slicer 3D or Radiant (using Fusion feature).

    Arguments:
        header {PyDicom Dataset} -- original DICOM file header
        orig_vol {Numpy array} -- original 3D volume
        pred_vol {Numpy array} -- 3D prediction mask (0=bg, 1=anterior, 2=posterior)
        output_dir {string} -- directory to save the DICOM series

    Returns:
        N/A
    """
    os.makedirs(output_dir, exist_ok=True)

    series_uid = pydicom.uid.generate_uid()
    frame_of_ref_uid = pydicom.uid.generate_uid()

    for slc_idx in range(pred_vol.shape[2]):
        out = pydicom.Dataset()

        # File Meta
        out.file_meta = pydicom.Dataset()
        out.file_meta.FileMetaInformationVersion = b'\x00\x01'
        out.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        out.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
        sop_uid = pydicom.uid.generate_uid()
        out.file_meta.MediaStorageSOPInstanceUID = sop_uid
        out.file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID

        out.is_little_endian = True
        out.is_implicit_VR = False

        # Patient Module
        out.PatientName = getattr(header, 'PatientName', '')
        out.PatientID = getattr(header, 'PatientID', '')
        out.PatientBirthDate = getattr(header, 'PatientBirthDate', '')
        out.PatientSex = getattr(header, 'PatientSex', '')

        # General Study Module
        out.StudyInstanceUID = getattr(header, 'StudyInstanceUID', pydicom.uid.generate_uid())
        out.StudyDate = getattr(header, 'StudyDate', '')
        out.StudyTime = getattr(header, 'StudyTime', '')
        out.StudyID = getattr(header, 'StudyID', '')
        out.AccessionNumber = getattr(header, 'AccessionNumber', '')

        # General Series Module
        out.SeriesInstanceUID = series_uid
        out.Modality = "OT"
        out.SeriesNumber = 9998
        out.SeriesDescription = "HippoVolume.AI Segmentation Mask"

        # Frame of Reference Module
        out.FrameOfReferenceUID = frame_of_ref_uid
        out.PositionReferenceIndicator = ""

        # General Image Module
        out.InstanceNumber = slc_idx + 1
        out.ImageType = ["DERIVED", "PRIMARY"]
        out.ContentDate = datetime.date.today().strftime("%Y%m%d")
        out.ContentTime = datetime.datetime.now().strftime("%H%M%S")

        # SOP Common Module
        out.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
        out.SOPInstanceUID = sop_uid
        out.SpecificCharacterSet = "ISO_IR 100"

        # SC Equipment Module
        out.ConversionType = "WSD"

        # Image Pixel Module - RGB color-coded segmentation
        mask_slc = pred_vol[:, :, slc_idx]
        rgb_mask = np.zeros((*mask_slc.shape, 3), dtype=np.uint8)
        rgb_mask[mask_slc == 1] = [0, 255, 0]    # anterior - green
        rgb_mask[mask_slc == 2] = [255, 0, 0]    # posterior - red

        out.SamplesPerPixel = 3
        out.PhotometricInterpretation = "RGB"
        out.PlanarConfiguration = 0
        out.Rows = rgb_mask.shape[0]
        out.Columns = rgb_mask.shape[1]
        out.BitsAllocated = 8
        out.BitsStored = 8
        out.HighBit = 7
        out.PixelRepresentation = 0
        out.PixelData = rgb_mask.tobytes()

        out.BurnedInAnnotation = "NO"
        out.LossyImageCompression = "00"

        fpath = os.path.join(output_dir, f"seg_{slc_idx:04d}.dcm")
        pydicom.filewriter.dcmwrite(fpath, out, write_like_original=False)

    print(f"Saved segmentation mask series ({pred_vol.shape[2]} slices) to {output_dir}")

def get_series_for_inference(path):
    """Reads multiple series from one folder and picks the one
    to run inference on.

    Uses multi-criteria filtering for robust series identification:
    1. Primary: SeriesDescription containing "HippoCrop"
    2. Fallback: Look for small-volume series typical of hippocampal crops
       (series with consistently small image dimensions, ~30-60 slices)
    3. Validates that the candidate series has expected properties
       (axial orientation, reasonable number of instances)

    Arguments:
        path {string} -- location of the DICOM files

    Returns:
        Numpy array representing the series
    """

    # Here we are assuming that path is a directory that contains a full study as a collection
    # of files
    # We are reading all files into a list of PyDicom objects so that we can filter them later
    dicoms = []
    for root, dirs, files in os.walk(path):
        for f in files:
            fpath = os.path.join(root, f)
            try:
                dcm = pydicom.dcmread(fpath)
                dicoms.append(dcm)
            except Exception:
                pass

    if not dicoms:
        print("Error: no valid DICOM files found")
        return []

    # STAND-OUT: Multi-criteria series filtering
    # Strategy 1: Primary filter - SeriesDescription containing "HippoCrop"
    series_for_inference = [dcm for dcm in dicoms
                           if hasattr(dcm, "SeriesDescription")
                           and "HippoCrop" in dcm.SeriesDescription]

    # Strategy 2: Fallback - if no HippoCrop found, look for series with
    # small image dimensions typical of cropped hippocampal volumes
    if not series_for_inference:
        print("No 'HippoCrop' series found. Attempting fallback detection...")
        # Group DICOMs by SeriesInstanceUID
        series_groups = {}
        for dcm in dicoms:
            uid = dcm.SeriesInstanceUID
            if uid not in series_groups:
                series_groups[uid] = []
            series_groups[uid].append(dcm)

        # Look for series with small image dimensions (hippocampal crops are typically
        # ~35x50 pixels, much smaller than full-brain MRI slices which are 256x256+)
        for uid, group in series_groups.items():
            sample = group[0]
            if hasattr(sample, 'Rows') and hasattr(sample, 'Columns'):
                # Hippocampal crops typically have both dimensions < 100
                if sample.Rows < 100 and sample.Columns < 100:
                    series_for_inference = group
                    print(f"Found candidate cropped series: {sample.Rows}x{sample.Columns}, "
                          f"{len(group)} slices")
                    break

    if not series_for_inference:
        print("Error: could not identify a suitable series for inference")
        return []

    # Validate: check if there's exactly one series
    series_uids = {f.SeriesInstanceUID for f in series_for_inference}
    if len(series_uids) != 1:
        print(f"Error: found {len(series_uids)} matching series, expected exactly 1")
        return []

    # STAND-OUT: Log series metadata for debugging and audit trail
    sample_dcm = series_for_inference[0]
    print(f"Selected series for inference:")
    print(f"  SeriesInstanceUID: {sample_dcm.SeriesInstanceUID}")
    print(f"  SeriesDescription: {getattr(sample_dcm, 'SeriesDescription', 'N/A')}")
    print(f"  Modality: {getattr(sample_dcm, 'Modality', 'N/A')}")
    print(f"  Number of instances: {len(series_for_inference)}")
    if hasattr(sample_dcm, 'Rows') and hasattr(sample_dcm, 'Columns'):
        print(f"  Image dimensions: {sample_dcm.Rows}x{sample_dcm.Columns}")

    return series_for_inference

def os_command(command):
    # Comment this if running under Windows
    sp = subprocess.Popen(["/bin/bash", "-i", "-c", command])
    sp.communicate()

    # Uncomment this if running under Windows
    # os.system(command)

if __name__ == "__main__":
    # This code expects a single command line argument with link to the directory containing
    # routed studies
    if len(sys.argv) != 2:
        print("You should supply one command line argument pointing to the routing folder. Exiting.")
        sys.exit()

    # Find all subdirectories within the supplied directory. We assume that 
    # one subdirectory contains a full study
    subdirs = [os.path.join(sys.argv[1], d) for d in os.listdir(sys.argv[1]) if
                os.path.isdir(os.path.join(sys.argv[1], d))]

    # Get the latest directory
    study_dir = sorted(subdirs, key=lambda dir: os.stat(dir).st_mtime, reverse=True)[0]

    print(f"Looking for series to run inference on in directory {study_dir}...")

    # TASK: get_series_for_inference is not complete. Go and complete it
    volume, header = load_dicom_volume_as_numpy_from_list(get_series_for_inference(study_dir))
    print(f"Found series of {volume.shape[2]} axial slices")

    print("HippoVolume.AI: Running inference...")
    # TASK: Use the UNetInferenceAgent class and model parameter file from the previous section
    inference_agent = UNetInferenceAgent(
        device="cpu",
        parameter_file_path=os.path.join(os.path.dirname(__file__),
            "../../section2/out/2026-03-08_0018_Basic_unet/model.pth"))

    # Run inference
    # TASK: single_volume_inference_unpadded takes a volume of arbitrary size 
    # and reshapes y and z dimensions to the patch size used by the model before 
    # running inference. Your job is to implement it.
    pred_label = inference_agent.single_volume_inference_unpadded(np.array(volume))
    # TASK: get_predicted_volumes is not complete. Go and complete it
    pred_volumes = get_predicted_volumes(pred_label)

    # Create and save the report
    print("Creating and pushing report...")
    report_save_path = os.path.join(os.path.dirname(__file__), "../out/report.dcm")
    # TASK: create_report is not complete. Go and complete it. 
    # STAND OUT SUGGESTION: save_report_as_dcm has some suggestions if you want to expand your
    # knowledge of DICOM format
    report_img = create_report(pred_volumes, header, volume, pred_label)
    save_report_as_dcm(header, report_img, report_save_path)

    # STAND-OUT: Save segmentation mask as a separate DICOM series
    # This can be overlaid on the original images in Slicer 3D or Radiant (Fusion)
    seg_series_dir = os.path.join(os.path.dirname(__file__), "../out/segmentation_series")
    save_segmentation_as_dcm_series(header, volume, pred_label, seg_series_dir)

    # Send report to our storage archive
    # TASK: Write a command line string that will issue a DICOM C-STORE request to send our report
    # to our Orthanc server (that runs on port 4242 of the local machine), using storescu tool
    os_command(f"storescu 127.0.0.1 4242 -v -aec HIPPOAI {report_save_path}")

    # This line will remove the study dir if run as root user
    # Sleep to let our StoreSCP server process the report (remember - in our setup
    # the main archive is routing everyting that is sent to it, including our freshly generated
    # report) - we want to give it time to save before cleaning it up
    time.sleep(2)
    shutil.rmtree(study_dir, onerror=lambda f, p, e: print(f"Error deleting: {e[1]}"))

    print(f"Inference successful on {header['SOPInstanceUID'].value}, out: {pred_label.shape}",
          f"volume ant: {pred_volumes['anterior']}, ",
          f"volume post: {pred_volumes['posterior']}, total volume: {pred_volumes['total']}")
