from pathlib import Path

from karcytics_sdk.plugin.theme_fallback import Colors
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHeaderView,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class ChannelManagerWidget(QWidget):
    """UI Component for adding images and assigning them to color channels."""

    channels_changed = pyqtSignal()
    new_image_loaded = pyqtSignal(str)

    def __init__(self, image_stack, parent=None):
        super().__init__(parent)
        self.image_stack = image_stack
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.btn_add = QPushButton("➕ Add Image Channel")
        self.btn_add.setStyleSheet(f"""
            QPushButton {{ background-color: {Colors.ACCENT_PRIMARY}; color: {Colors.BG_DARKEST}; font-weight: bold; padding: 8px; border-radius: 4px; }}
            QPushButton:hover {{ border: 1px solid white; }}
        """)
        self.btn_add.clicked.connect(self._on_add_channel)
        layout.addWidget(self.btn_add)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["File", "Color"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.setStyleSheet(
            f"background: {Colors.BG_DARKEST}; color: {Colors.FG_PRIMARY}; gridline-color: {Colors.BORDER};"
        )
        self.table.verticalHeader().hide()

        # --- NEW: Listen for user edits in the table ---
        self.table.itemChanged.connect(self._on_item_changed)

        layout.addWidget(self.table)

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

        final_path = Path(
            path
        )  # Make sure you have 'from pathlib import Path' at the top of your file

        # 3. Handle Workspace Integration (Copying external files to 'assets')
        if pm:
            try:
                is_in_workspace = pm.assets_dir.resolve() in final_path.resolve().parents

                if not is_in_workspace:
                    reply = QMessageBox.question(
                        self,
                        "Copy to Workspace?",
                        f"The image '{final_path.name}' is outside the project folder.\n\n"
                        "Would you like to copy it into the project's 'assets' folder for safe keeping and portability?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.Yes,
                    )
                    copy_to_workspace = reply == QMessageBox.StandardButton.Yes
                else:
                    copy_to_workspace = False

                file_hash = pm.add_image(final_path, copy_to_workspace)
                resolved_path = pm.get_asset_path(file_hash)
                if resolved_path:
                    final_path = resolved_path

            except Exception as e:
                QMessageBox.warning(self, "Asset Error", f"Failed to add asset to project:\n{e}")
                return  # Abort if we can't manage the asset safely

        # Convert back to a standard string for your CytoMetrics logic
        file_path_str = str(final_path)

        # 4. CYTOMETRICS LOAD LOGIC
        name = final_path.name
        default_colors = ["gray", "green", "magenta", "blue", "red"]
        color = default_colors[len(self.image_stack.channels) % len(default_colors)]

        added_channels = self.image_stack.add_channel(file_path_str, name, color)

        if added_channels:
            for ch_name, ch_color in added_channels:
                self._add_row_to_ui(ch_name, ch_color)

                for c in self.image_stack.channels:
                    if c.name == ch_name:
                        c.path = file_path_str

            # Emit the signals using the new, safely managed path
            self.new_image_loaded.emit(file_path_str)
            self.channels_changed.emit()

    def _add_row_to_ui(self, name: str, current_color: str):
        # Block signals so creating the row doesn't trigger a fake "user edit"
        self.table.blockSignals(True)

        row = self.table.rowCount()
        self.table.insertRow(row)

        item = QTableWidgetItem(name)
        item.setToolTip(name)
        self.table.setItem(row, 0, item)

        combo = QComboBox()
        combo.addItems(["gray", "magenta", "green", "blue", "red", "cyan", "yellow"])
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
        if col == 0 and row < len(self.image_stack.channels):
            new_name = item.text()
            # 1. Update the underlying data model
            self.image_stack.channels[row].name = new_name
            # 2. Update the hover tooltip to match
            item.setToolTip(new_name)
            # 3. Tell the rest of the app to refresh!
            self.channels_changed.emit()

    def _on_color_changed(self, row: int, new_color: str):
        if row < len(self.image_stack.channels):
            self.image_stack.channels[row].color = new_color
            self.channels_changed.emit()

    def clear_ui(self):
        """Safely removes all channel rows from the UI table."""
        # Because you are using a QTableWidget, clearing it is this easy!
        self.table.setRowCount(0)

    def cleanup(self) -> None:
        """Release UI resources. Called when the plugin panel is closed."""
        self.table.blockSignals(True)
        self.clear_ui()
        self.image_stack = None  # Release reference to data model
