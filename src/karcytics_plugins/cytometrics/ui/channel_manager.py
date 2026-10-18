from pathlib import Path

from karcytics_sdk.plugin import (
    BioCaptionLabel,
    BioComboBox,
    BioDoubleSpinBox,
    BioTableWidget,
    PrimaryButton,
    SecondaryButton,
    SubtitleLabel,
)
from karcytics_sdk.plugin.dialogs import ask_yes_no, show_warning
from karcytics_sdk.plugin.theme_fallback import Colors
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..analysis.model import CHANNEL_COLORS, ImageSet


class ChannelManagerWidget(QWidget):
    """UI Component for adding images and assigning them to color channels."""

    channels_changed = Signal()
    new_image_loaded = Signal(str)

    def __init__(self, image_set: ImageSet, parent=None):
        super().__init__(parent)
        self.image_set: ImageSet | None = image_set
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.btn_add = PrimaryButton("➕ Add Image Channel")
        self.btn_add.clicked.connect(self._on_add_channel)
        layout.addWidget(self.btn_add)

        self.table = BioTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["File", "Color"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.setStyleSheet(
            f"background: {Colors.BG_DARKEST}; color: {Colors.FG_PRIMARY}; gridline-color: {Colors.BORDER};"
        )
        self.table.verticalHeader().hide()

        self.table.setToolTip("Tick to show a channel; double-click its name to rename it.")
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.itemSelectionChanged.connect(self._on_row_selected)
        layout.addWidget(self.table)

        # Display contrast of the selected channel (the canvas only; analysis uses all data).
        layout.addWidget(SubtitleLabel("Display contrast"))
        self.lbl_contrast = BioCaptionLabel("Select a channel to adjust how it's displayed.")
        layout.addWidget(self.lbl_contrast)
        row = QHBoxLayout()
        self.lbl_low, self.lbl_high = QLabel("Black:"), QLabel("White:")
        self.spin_low, self.spin_high = BioDoubleSpinBox(), BioDoubleSpinBox()
        for spin in (self.spin_low, self.spin_high):
            spin.setDecimals(0)
            spin.setRange(-1e9, 1e9)
            spin.setEnabled(False)
            spin.valueChanged.connect(self._on_contrast_edited)
        self.btn_auto = SecondaryButton("Auto")
        self.btn_auto.setToolTip("Stretch from the 0.1st to the 99.9th percentile.")
        self.btn_auto.setEnabled(False)
        self.btn_auto.clicked.connect(self._on_auto_contrast)
        for w in (self.lbl_low, self.spin_low, self.lbl_high, self.spin_high, self.btn_auto):
            row.addWidget(w)
        layout.addLayout(row)

    def _apply_theme_styles(self) -> None:
        """Re-applies theme-aware styles when the active theme changes.

        ``self.table`` and each row's color-picker combo bake ``Colors.*``
        into their stylesheet at construction time (self.btn_add, an SDK
        PrimaryButton, already re-styles itself), so they need an explicit
        refresh here.
        """
        self.table.setStyleSheet(
            f"background: {Colors.BG_DARKEST}; color: {Colors.FG_PRIMARY}; gridline-color: {Colors.BORDER};"
        )
        combo_style = f"background: {Colors.BG_DARK}; color: {Colors.FG_PRIMARY};"
        for label in (self.lbl_low, self.lbl_high):
            label.setStyleSheet(f"color: {Colors.FG_PRIMARY};")
        for row in range(self.table.rowCount()):
            combo = self.table.cellWidget(row, 1)
            if combo is not None:
                combo.setStyleSheet(combo_style)

    def prompt_add_channel(self):
        """Public entry point to open the add-image file dialog, for callers outside this widget."""
        self._on_add_channel()

    def _on_add_channel(self):
        # 1. Grab the Project Manager from the main application window
        main_win = self.window()
        pm = getattr(main_win, "project_manager", None)
        default_dir = str(pm.project_dir) if pm else ""

        # 2. Open the File Dialog rooted in the project directory
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image Channel",
            default_dir,
            "Image Files (*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;All Files (*)",
        )
        if not path:
            return

        self.ingest_path(path)

    def ingest_path(self, path: str):
        """Add a single image file as a channel. Shared by the file dialog and drag-and-drop."""
        main_win = self.window()
        pm = getattr(main_win, "project_manager", None)

        final_path = Path(path)

        # 3. Handle Workspace Integration (Copying external files to 'assets')
        if pm:
            try:
                is_in_workspace = pm.assets_dir.resolve() in final_path.resolve().parents

                if not is_in_workspace:
                    copy_to_workspace = ask_yes_no(
                        self,
                        "Copy to Workspace?",
                        f"The image '{final_path.name}' is outside the project folder.\n\n"
                        "Would you like to copy it into the project's 'assets' folder for safe keeping and portability?",
                    )
                else:
                    copy_to_workspace = False

                file_hash = pm.add_image(final_path, copy_to_workspace)
                resolved_path = pm.get_asset_path(file_hash)
                if resolved_path:
                    final_path = Path(resolved_path)

            except Exception as e:
                show_warning(self, "Asset Error", f"Failed to add asset to project:\n{e}")
                return  # Abort if we can't manage the asset safely

        assert self.image_set is not None
        palette = ["gray", "green", "magenta", "blue", "red"]
        color = palette[len(self.image_set.channels) % len(palette)]
        try:
            added = self.image_set.add_file(final_path, final_path.name, color)
        except (OSError, ValueError) as e:  # includes UnsupportedImageError
            show_warning(self, "Can't Add Image", f"Could not read '{final_path.name}':\n{e}")
            return

        for ch in added:
            self._add_row_to_ui(ch.name, ch.color)
        self.new_image_loaded.emit(str(final_path))
        self.channels_changed.emit()

    def ingest_paths(self, paths: list[str]):
        """Add multiple image files as channels, e.g. from a drag-and-drop event."""
        for path in paths:
            self.ingest_path(path)

    def _add_row_to_ui(self, name: str, current_color: str):
        # Block signals so creating the row doesn't trigger a fake "user edit"
        self.table.blockSignals(True)

        row = self.table.rowCount()
        self.table.insertRow(row)

        item = QTableWidgetItem(name)
        item.setToolTip(name)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        visible = (
            self.image_set.channels[row].visible
            if self.image_set and row < len(self.image_set.channels)
            else True
        )
        item.setCheckState(Qt.CheckState.Checked if visible else Qt.CheckState.Unchecked)
        self.table.setItem(row, 0, item)

        combo = BioComboBox()
        combo.addItems(CHANNEL_COLORS)
        combo.setCurrentText(current_color)
        combo.currentTextChanged.connect(lambda text, r=row: self._on_color_changed(r, text))
        combo.setStyleSheet(f"background: {Colors.BG_DARK}; color: {Colors.FG_PRIMARY};")
        self.table.setCellWidget(row, 1, combo)

        self.table.blockSignals(False)

    # --- NEW: Save the edited name to memory ---
    def _on_item_changed(self, item):
        row = item.row()
        col = item.column()

        # We only care if they edited the Name column (Column 0)
        if col == 0 and self.image_set and row < len(self.image_set.channels):
            channel = self.image_set.channels[row]
            channel.name = item.text()
            channel.visible = item.checkState() == Qt.CheckState.Checked
            item.setToolTip(channel.name)
            self.channels_changed.emit()

    def _on_color_changed(self, row: int, new_color: str):
        if self.image_set and row < len(self.image_set.channels):
            self.image_set.channels[row].color = new_color
            self.channels_changed.emit()

    def _selected_channel(self):
        rows = {i.row() for i in self.table.selectedIndexes()}
        if len(rows) != 1 or not self.image_set:
            return None
        row = rows.pop()
        return self.image_set.channels[row] if row < len(self.image_set.channels) else None

    def _on_row_selected(self) -> None:
        channel = self._selected_channel()
        for w in (self.spin_low, self.spin_high, self.btn_auto):
            w.setEnabled(channel is not None)
        if channel is None:
            self.lbl_contrast.setText("Select a channel to adjust how it's displayed.")
            return
        self._show_range(channel)

    def _show_range(self, channel) -> None:
        lo, hi = channel.display_range or channel.auto_range()
        full_lo, full_hi = channel.full_range()
        mode = "manual" if channel.display_range else "auto"
        self.lbl_contrast.setText(
            f"'{channel.name}' ({channel.data.dtype}, {mode}). Values from {full_lo:g} to "
            f"{full_hi:g}; only the display changes."
        )
        for spin, value in ((self.spin_low, lo), (self.spin_high, hi)):
            spin.blockSignals(True)
            spin.setRange(full_lo, full_hi)
            spin.setValue(value)
            spin.blockSignals(False)

    def _on_contrast_edited(self, *_args) -> None:
        channel = self._selected_channel()
        if channel is None:
            return
        lo, hi = self.spin_low.value(), self.spin_high.value()
        if hi > lo:
            channel.display_range = (lo, hi)
            self._show_range(channel)
            self.channels_changed.emit()

    def _on_auto_contrast(self) -> None:
        channel = self._selected_channel()
        if channel is None:
            return
        channel.display_range = None
        self._show_range(channel)
        self.channels_changed.emit()

    def clear_ui(self):
        """Safely removes all channel rows from the UI table."""
        self.table.setRowCount(0)

    def refresh(self) -> None:
        """Rebuild the table from the image set, e.g. after a workflow is restored."""
        self.clear_ui()
        for ch in self.image_set.channels if self.image_set else []:
            self._add_row_to_ui(ch.name, ch.color)

    def cleanup(self) -> None:
        """Release UI resources. Called when the plugin panel is closed."""
        self.table.blockSignals(True)
        self.clear_ui()
        self.image_set = None  # Release reference to data model
