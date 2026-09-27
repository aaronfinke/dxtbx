from __future__ import annotations

import logging
import os
import pathlib
import xml.etree.ElementTree as ET

import h5py
import numpy as np

import cctbx.array_family.flex as flex

import dxtbx_flumpy as flumpy
from dxtbx import IncorrectFormatError
from dxtbx.format.FormatESSNMX import FormatESSNMX
from dxtbx.model.beam import BeamFactory, PolychromaticBeam, Probe
from dxtbx.model.detector import Detector
from dxtbx.model.goniometer import GoniometerFactory
from dxtbx.model.scan import Scan, ScanFactory
from dxtbx.model.tof_helpers import InstrumentDefinitionReader

logger = logging.getLogger(__name__)


class FormatMANDI_NXLaueTOF(FormatESSNMX):
    """
    Reads MANDI data histogrammed into an NXlauetof file by essmandi-reduce
    (ess.mandi.workflows), so that binning does not have to be done through
    FormatMANDI.write_histogram_data.

    Geometry is taken from entry/instrument/IDF, the Mantid instrument
    definition that essmandi-reduce copies into the reduced file, and is read
    with the same InstrumentDefinitionReader that FormatMANDI uses.  The
    exported IDF is byte-identical to the one in the original event file, so the
    detector this builds agrees with FormatMANDI by construction rather than by
    correcting the NeXus fields.

    That matters because the NeXus fields in these files do not match
    FormatMANDI: fast and slow axes are exchanged, x_pixel_size/y_pixel_size
    say 0.6283 mm where FormatMANDI and the ISAW DetCal calibration both say
    0.6180 mm, and "first_pixel_position" is the centre of the first pixel
    rather than its corner.  They are only used here as a cross-check.

    essmandi-reduce writes entry/instrument/name as "NMX" rather than "MANDI",
    so FormatESSNMX.understand also matches these files.  This class therefore
    derives from it, which makes dxtbx's registry prefer this one, and
    discriminates on the detector names: MANDI writes bank<N>, NMX writes
    detector_panel_<N>.  None of FormatESSNMX's NMX-specific geometry is used -
    every method that touches geometry or data is overridden here.

    This is still the nominal geometry.  Individual MANDI panels sit up to
    12 px out of place, which no amount of refinement corrects, so a DetCal
    calibration has to be applied afterwards - currently by
    xia2.laue_tof detector_calibration=, which applies it at import.
    """

    _PIXEL_SIZE = 0.618  # (mm) FormatMANDI and DetCal agree; the file says 0.6283

    def __init__(self, image_file, **kwargs) -> None:
        if not FormatMANDI_NXLaueTOF.understand(image_file):
            raise IncorrectFormatError(self, image_file)
        super().__init__(image_file, **kwargs)
        self._nxs_file = h5py.File(image_file, "r")
        # FormatESSNMX's __init__ does not chain to Format.__init__, so
        # self._image_file is never set; keep the path here instead.
        self._nxlauetof_path = image_file
        self.xml_reader = InstrumentDefinitionReader()
        self.xml_file = ET.fromstring(
            _as_str(self._nxs_file["entry/instrument/IDF"][()])
        )

    def __del__(self) -> None:
        try:
            self._nxs_file.close()
        except Exception:
            pass

    @staticmethod
    def understand(image_file: str) -> bool:
        try:
            with h5py.File(image_file, "r") as handle:
                entry = handle["entry"]
                if _as_str(entry.attrs.get("NX_class")) != "NXlauetof":
                    return False
                # entry/instrument/name says "NMX" even for MANDI data, so the
                # instrument definition is used instead: its name attribute is
                # correct, and it is what the geometry is read from anyway.
                if "IDF" not in entry["instrument"]:
                    return False
                idf = ET.fromstring(_as_str(entry["instrument/IDF"][()]))
                if idf.get("name") != "MANDI":
                    return False
                banks = [
                    key
                    for key, group in entry["instrument"].items()
                    if key.startswith("bank")
                    and isinstance(group, h5py.Group)
                    and "data" in group
                ]
                if not banks:
                    return False
                return entry[f"instrument/{banks[0]}/data"].ndim == 3
        except (OSError, KeyError, TypeError, AttributeError):
            return False

    # ---------------------------------------------------------------- panels

    def _panel_groups(self) -> list[tuple[str, h5py.Group]]:
        """(name, group) per detector, ordered by bank number as FormatMANDI is."""
        inst = self._nxs_file["entry/instrument"]
        out = []
        for key in inst:
            group = inst[key]
            if not isinstance(group, h5py.Group) or "data" not in group:
                continue
            digits = "".join(c for c in key if c.isdigit())
            if digits:
                out.append((int(digits), key, group))
        out.sort()
        return [(key, group) for _, key, group in out]

    def _get_image_size(self) -> tuple[int, int]:
        _, group = self._panel_groups()[0]
        shape = group["data"].shape
        # stored (fast, slow, tof); a dxtbx image size is (fast, slow)
        return (int(shape[0]), int(shape[1]))

    def get_detector(self, index: int = None) -> Detector:
        """As FormatMANDI, from the same instrument definition."""
        panel_names = self.xml_reader.get_panel_names(self.xml_file)
        num_panels = self.xml_reader.get_num_panels(self.xml_file)
        origins, fast_axes, slow_axes = self.xml_reader.get_dials_detector_geometry(
            self.xml_file
        )
        image_size = self._get_image_size()
        pixel_size = self._get_pixel_size()
        projections = self._panel_projections_2d(num_panels, image_size)
        detector = Detector()
        root = detector.hierarchy()
        for i in range(num_panels):
            group = root.add_group()
            panel = group.add_panel()
            panel.set_type("SENSOR_PAD")
            panel.set_name(panel_names[i])
            panel.set_image_size(image_size)
            panel.set_trusted_range((-1, 100000))
            panel.set_pixel_size(pixel_size)
            panel.set_local_frame(fast_axes[i], slow_axes[i], origins[i])
            panel.set_gain(1.0)
            r, t = projections[i]
            panel.set_projection_2d(tuple(map(int, r)), tuple(map(int, t)))
        return detector

    def _get_pixel_size(self) -> tuple[float, float]:
        """(mm).  The file's x_pixel_size/y_pixel_size say 0.6283, which
        disagrees with both FormatMANDI and the DetCal calibration; the
        calibration is the one validated against Mantid's peak positions."""
        return (self._PIXEL_SIZE, self._PIXEL_SIZE)

    @staticmethod
    def _panel_projections_2d(n_panels, image_size):
        """A plain grid; only used for 2D display, not for any calculation."""
        p_w, p_h = image_size[0] + 10, image_size[1] + 10
        cols = int(np.ceil(np.sqrt(n_panels)))
        return {
            i: ((1, 0, 0, 1), (p_h * (i // cols), p_w * (i % cols)))
            for i in range(n_panels)
        }

    # ------------------------------------------------------------ scan, beam

    _TO_US = {"ns": 1e-3, "us": 1.0, "ms": 1e3, "s": 1e6}

    def _to_us(self, dataset) -> np.ndarray:
        """Values in us.  essmandi-reduce defaults to ns
        (--result-time-bin-unit), and FormatMANDI's scan is in us."""
        values = np.asarray(dataset[...], dtype=float)
        unit = _as_str(dataset.attrs.get("units")) or "us"
        try:
            return values * self._TO_US[unit]
        except KeyError:
            raise ValueError(
                f"Unrecognised time unit {unit!r} on {dataset.name}; "
                f"expected one of {sorted(self._TO_US)}"
            ) from None

    def _tof_bin_edges(self) -> np.ndarray:
        """Bin edges (us).

        Deliberately not called _time_channel_bins: some versions of
        FormatESSNMX set an instance attribute of that name in __init__, which
        would shadow this method.
        """
        _, group = self._panel_groups()[0]
        if "original_time_edges" in group:
            return self._to_us(group["original_time_edges"])
        centres = self._to_us(group["time_of_flight"])
        step = np.diff(centres)
        return np.concatenate(
            [
                [centres[0] - step[0] / 2],
                centres[:-1] + step / 2,
                [centres[-1] + step[-1] / 2],
            ]
        )

    def _get_time_of_flight(self) -> tuple[float, ...]:
        """Bin centres (us), as FormatMANDI reports them."""
        bins = self._tof_bin_edges()
        return tuple(float(bins[i] + bins[i + 1]) * 0.5 for i in range(len(bins) - 1))

    def get_num_images(self) -> int:
        return len(self._get_time_of_flight())

    def get_scan(self, index=None) -> Scan:
        return ScanFactory.make_scan_from_properties(
            image_range=(1, self.get_num_images()),
            properties={"time_of_flight": self._get_time_of_flight()},
        )

    def get_beam(self, index: int = None) -> PolychromaticBeam:
        return BeamFactory.make_polychromatic_beam(
            direction=(0, 0, -1),
            sample_to_source_distance=30000,  # (mm) as FormatMANDI
            probe=Probe.neutron,
            wavelength_range=(2.0, 4.0),
        )

    # ------------------------------------------------------------ goniometer

    _SOURCE_DIR_ENV = "DXTBX_MANDI_SOURCE_DIR"

    def _source_event_file(self) -> pathlib.Path | None:
        """The original MANDI event file, for metadata the reduced file lacks.

        essmandi-reduce does not carry the goniometer angles over: entry/sample
        holds placeholders (crystal_rotation is [0, 0, 0]) and there are no
        DASlogs, so the reduced file cannot say which of several goniometer
        settings it came from.  The angles are therefore read from the original
        event file, which is located by, in order:

          1. $DXTBX_MANDI_SOURCE_DIR, used directly if it names a file,
             otherwise <that directory>/<stem>.nxs.h5
          2. <stem>.nxs.h5 beside the reduced file
          3. ../DataFiles/<stem>.nxs.h5

        This is a stopgap.  Once the angles are written into the reduced file
        it should be dropped in favour of reading them from there.
        """
        reduced = pathlib.Path(self._nxlauetof_path)
        stem = reduced.name
        for suffix in (".nxs", ".h5", ".nxs.h5"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        name = f"{stem}.nxs.h5"
        candidates = []
        env = os.environ.get(self._SOURCE_DIR_ENV)
        if env:
            env_path = pathlib.Path(env)
            # accept either the event file itself or the directory holding it,
            # since the reduced file need not be named after its run
            if env_path.is_file():
                return env_path
            candidates.append(env_path / name)
        candidates.append(reduced.parent / name)
        candidates.append(reduced.parent.parent / "DataFiles" / name)
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None

    def get_goniometer(self, index: int = None):
        """As FormatMANDI: phi, then chi, then omega, about the same axes."""
        rotation_axis_phi = (0.0, 1.0, 0.0)
        rotation_axis_omega = (0.0, 1.0, 0.0)
        rotation_axis_chi = (0.0, 0.0, 1.0)
        fixed_rotation = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

        phi, chi, omega = self._goniometer_angles()
        goniometer = GoniometerFactory.make_goniometer(
            rotation_axis_phi, fixed_rotation
        )
        goniometer.rotate_around_origin(rotation_axis_phi, phi)
        goniometer.rotate_around_origin(rotation_axis_chi, chi)
        goniometer.rotate_around_origin(rotation_axis_omega, omega)
        return goniometer

    def _goniometer_angles(self) -> tuple[float, float, float]:
        """(phi, chi, omega) in degrees."""
        sample = self._nxs_file["entry/sample"]
        if "crystal_rotation" in sample:
            values = np.asarray(sample["crystal_rotation"][...], dtype=float)
            if values.size == 3 and np.any(values != 0.0):
                # written as (phi, chi, omega) to match FormatMANDI's order
                return float(values[0]), float(values[1]), float(values[2])

        source = self._source_event_file()
        if source is None:
            raise RuntimeError(
                f"{pathlib.Path(self._nxlauetof_path).name} carries no goniometer "
                "angles: entry/sample/crystal_rotation is all zeros and there "
                "are no DASlogs, so essmandi-reduce has not copied them over. "
                "The original event file is needed to supply them; put it "
                "beside this file, or in a sibling DataFiles directory, or set "
                f"${self._SOURCE_DIR_ENV} to the directory holding it."
            )
        with h5py.File(source, "r") as handle:
            angles = tuple(
                float(handle[f"entry/DASlogs/{name}/average_value"][0])
                for name in ("phi", "chi", "omega")
            )
        logger.debug("read goniometer angles %s from %s", angles, source)
        return angles

    # ------------------------------------------------------------------ data

    def get_raw_data(self, index: int) -> tuple[flex.int]:
        raw_data = []
        for _, group in self._panel_groups():
            # stored (fast, slow, tof); dxtbx wants (slow, fast) per image
            frame = group["data"][:, :, index : index + 1]
            frame = np.ascontiguousarray(
                np.asarray(frame, dtype=np.int32).reshape(frame.shape[:2]).T
            )
            raw_data.append(flumpy.from_numpy(frame))
        return tuple(raw_data)


def _as_str(value) -> str:
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode()
    return str(value) if value is not None else ""


if __name__ == "__main__":
    import sys

    for arg in sys.argv[1:]:
        print(FormatMANDI_NXLaueTOF.understand(arg))
