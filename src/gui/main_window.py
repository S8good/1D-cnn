from __future__ import annotations

import sys
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ..core.digital_twin_service import DigitalTwinService


class LSPRDigitalTwinMainWindow(QMainWindow):
    def __init__(self, service: Optional[DigitalTwinService] = None):
        super().__init__()
        self.setWindowTitle("LSPR Digital Twin Visualizer")
        self.resize(1200, 700)

        self.service = service or DigitalTwinService()

        self.conc_label: QLabel
        self.peak_label: QLabel
        self.shift_label: QLabel
        self.amp_label: QLabel
        self.ai_conc_label: QLabel
        self.slider: QSlider
        self.figure: Figure
        self.canvas: FigureCanvas
        self.ax = None

        self.init_ui()
        self.update_plot(5.0)

    def init_ui(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        control_panel = QFrame()
        control_panel.setFixedWidth(360)
        control_layout = QVBoxLayout(control_panel)

        slider_group = QGroupBox("Target Concentration Simulation")
        slider_layout = QVBoxLayout()
        self.conc_label = QLabel("Current concentration: 5.0 ng/ml")
        self.conc_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #E74C3C;")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(1000)
        self.slider.setValue(50)
        self.slider.valueChanged.connect(self.slider_changed)
        slider_layout.addWidget(self.conc_label)
        slider_layout.addWidget(self.slider)
        slider_group.setLayout(slider_layout)

        data_group = QGroupBox("Realtime Physical Features")
        data_layout = QVBoxLayout()
        self.peak_label = QLabel("Predicted peak wavelength: -- nm")
        self.shift_label = QLabel("Predicted redshift Δλ: -- nm")
        self.amp_label = QLabel("Predicted amplitude A: -- a.u.")
        for lbl in [self.peak_label, self.shift_label, self.amp_label]:
            lbl.setStyleSheet("font-family: Consolas; font-size: 14px; padding: 5px;")
            data_layout.addWidget(lbl)
        data_group.setLayout(data_layout)

        ai_group = QGroupBox("Full-Spectrum AI Tools")
        ai_layout = QVBoxLayout()
        self.btn_inverse = QPushButton("Import spectrum and infer concentration")
        self.btn_inverse.setStyleSheet(
            "background-color: #2ECC71; color: white; font-weight: bold; border-radius: 4px; padding: 8px;"
        )
        self.btn_inverse.clicked.connect(self.inverse_concentration)
        self.btn_spec2spec = QPushButton("Import spectrum and predict spectrum")
        self.btn_spec2spec.setStyleSheet(
            "background-color: #8E44AD; color: white; font-weight: bold; border-radius: 4px; padding: 8px;"
        )
        self.btn_spec2spec.clicked.connect(self.predict_spectrum_from_input)
        self.ai_conc_label = QLabel("AI inferred concentration: -- ng/ml")
        self.ai_conc_label.setStyleSheet("font-size: 14px; color: #27AE60; font-weight: bold;")
        ai_layout.addWidget(self.btn_inverse)
        ai_layout.addWidget(self.btn_spec2spec)
        ai_layout.addWidget(self.ai_conc_label)
        ai_group.setLayout(ai_layout)

        control_layout.addWidget(slider_group)
        control_layout.addWidget(data_group)
        control_layout.addWidget(ai_group)
        control_layout.addStretch()

        canvas_group = QGroupBox("LSPR Spectrum View")
        canvas_layout = QVBoxLayout()
        self.figure = Figure(figsize=(8, 6), dpi=100, facecolor="#F5F7FA")
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        canvas_layout.addWidget(self.canvas)
        canvas_group.setLayout(canvas_layout)

        main_layout.addWidget(control_panel)
        main_layout.addWidget(canvas_group)

    def slider_changed(self, value: int) -> None:
        conc = value / 10.0
        self.conc_label.setText(f"Current concentration: {conc:.1f} ng/ml")
        self.update_plot(conc)

    def update_plot(self, conc: float) -> None:
        context = self.service.build_plot_context(conc)
        pred = context.prediction

        self.peak_label.setText(f"Predicted peak wavelength: {pred.peak_wavelength:.2f} nm")
        self.shift_label.setText(f"Predicted redshift Δλ: +{pred.delta_lambda:.2f} nm")
        self.amp_label.setText(f"Predicted amplitude A: {pred.peak_intensity:.4f} a.u.")

        self.ax.clear()
        self.ax.plot(
            context.wavelengths,
            context.bsa_spectrum,
            color="#7F8C8D",
            linewidth=2,
            label="BSA baseline (pre)",
        )
        self.ax.plot(
            context.wavelengths,
            context.physical_spectrum,
            color="#E74C3C",
            linestyle="--",
            linewidth=2.5,
            label="Physical formula (post)",
        )

        if context.ai_spectrum_aligned is not None:
            self.ax.plot(
                context.ai_wavelengths,
                context.ai_spectrum_aligned,
                color="#9B59B6",
                linewidth=2.5,
                alpha=0.8,
                label="AI digital twin (generator, aligned)",
            )

        self.ax.set_title(f"LSPR Spectral Shift Simulation (CEA = {conc:.1f} ng/ml)", fontsize=14, fontweight="bold")
        self.ax.set_xlabel("Wavelength (nm)")
        self.ax.set_ylabel("Absorbance (a.u.)")
        self.ax.set_xlim(520, 700)
        self.ax.grid(True, linestyle=":", alpha=0.6)
        self.ax.legend(loc="upper right", fontsize=9)
        self.figure.tight_layout()
        self.canvas.draw()

    def inverse_concentration(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Choose measured spectrum file",
            "",
            "Spectrum Files (*.csv *.txt *.tsv *.xlsx *.xls);;CSV Files (*.csv);;Text Files (*.txt *.tsv);;Excel Files (*.xlsx *.xls);;All Files (*)",
        )

        try:
            conc = self.slider.value() / 10.0
            inference = self.service.infer_concentration_from_file(file_name or None, fallback_concentration=conc)
            pred_conc = float(inference["pred_concentration"])
            report = inference["report"]
            self.ai_conc_label.setText(f"AI inferred concentration: {report['reported_text']}")
            QMessageBox.information(
                self,
                "Inference done",
                (
                    f"Prediction mode: {report['mode']}\n"
                    f"Reported result: {report['reported_text']}\n"
                    f"Raw model output: {pred_conc:.4f} ng/ml"
                ),
            )
        except Exception as e:
            QMessageBox.warning(self, "Import failed", f"Unable to run concentration inference: {str(e)}")

    def predict_spectrum_from_input(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Choose input spectrum file",
            "",
            "Spectrum Files (*.csv *.txt *.tsv *.xlsx *.xls);;CSV Files (*.csv);;Text Files (*.txt *.tsv);;Excel Files (*.xlsx *.xls);;All Files (*)",
        )
        if not file_name:
            return

        try:
            result = self.service.predict_spectrum_from_file(file_name)
            pred_conc = float(result["pred_concentration"])
            report_text = result.get("reported_text", f"{pred_conc:.4f} ng/ml")

            wl = result["wavelengths"]
            input_spec = result["input_resampled"]
            pred_spec = result["pred_spectrum"]

            self.ax.clear()
            self.ax.plot(wl, input_spec, color="#1F77B4", linewidth=2.0, label="Input spectrum")
            self.ax.plot(wl, pred_spec, color="#D62728", linestyle="--", linewidth=2.2, label="Predicted spectrum")
            self.ax.set_title(f"Spectrum-to-Spectrum Prediction ({report_text})", fontsize=14, fontweight="bold")
            self.ax.set_xlabel("Wavelength (nm)")
            self.ax.set_ylabel("Absorbance (a.u.)")
            self.ax.grid(True, linestyle=":", alpha=0.6)
            self.ax.legend(loc="upper right", fontsize=9)
            self.figure.tight_layout()
            self.canvas.draw()

            self.ai_conc_label.setText(f"AI inferred concentration: {report_text}")
            QMessageBox.information(
                self,
                "Prediction done",
                (
                    "Spectrum-to-spectrum prediction finished.\n\n"
                    f"Reported result: {report_text}\n"
                    f"Raw model output: {pred_conc:.4f} ng/ml"
                ),
            )
        except Exception as e:
            QMessageBox.warning(self, "Prediction failed", f"Unable to run spectrum-to-spectrum prediction: {str(e)}")


def run_app(argv: Optional[list[str]] = None) -> int:
    qt_argv = argv if argv is not None else sys.argv
    app = QApplication(qt_argv)
    window = LSPRDigitalTwinMainWindow()
    window.show()
    return app.exec_()
