import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QComboBox, QSlider, QGroupBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from atypic.effects import RollPixelsEffect, RandomRollPixelsEffect, ColorValueEffect, ColorChannelSplitEffect, CorruptionEffect, SortEffect, ColorPaletteReductionEffect
from atypic.mask import Mask

class ImageEffectGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Atypic Effects GUI')
        self.image = None
        self.mask_obj = None
        self.mask_type = None
        self.mask_pos = None
        self.dragging = False
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.img_label = QLabel('No image loaded')
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setMouseTracking(True)
        self.img_label.mousePressEvent = self.mouse_press_event
        self.img_label.mouseMoveEvent = self.mouse_move_event
        self.img_label.mouseReleaseEvent = self.mouse_release_event
        layout.addWidget(self.img_label)

        btn_layout = QHBoxLayout()
        self.upload_btn = QPushButton('Upload Image')
        self.upload_btn.clicked.connect(self.upload_image)
        btn_layout.addWidget(self.upload_btn)
        layout.addLayout(btn_layout)

        # Mask controls
        mask_group = QGroupBox('Mask')
        mask_layout = QHBoxLayout()
        self.mask_combo = QComboBox()
        self.mask_combo.addItems(['Full', 'Rectangle', 'Circle', 'Polygon', 'Ellipse', 'Checkerboard', 'Stripe'])
        mask_layout.addWidget(self.mask_combo)
        self.apply_mask_btn = QPushButton('Apply Mask')
        self.apply_mask_btn.clicked.connect(self.apply_mask)
        mask_layout.addWidget(self.apply_mask_btn)
        mask_group.setLayout(mask_layout)
        layout.addWidget(mask_group)

        # Effect controls
        effect_group = QGroupBox('Effect')
        effect_layout = QHBoxLayout()
        self.effect_combo = QComboBox()
        self.effect_combo.addItems(['RollPixels', 'RandomRollPixels', 'ColorValue', 'ColorChannelSplit', 'Corruption', 'Sort', 'PaletteReduction'])
        self.effect_combo.currentIndexChanged.connect(self.update_preview_on_effect_change)
        effect_layout.addWidget(self.effect_combo)
        self.apply_effect_btn = QPushButton('Apply Effect')
        self.apply_effect_btn.clicked.connect(self.apply_effect)
        effect_layout.addWidget(self.apply_effect_btn)
        effect_group.setLayout(effect_layout)
        layout.addWidget(effect_group)
        self.setLayout(layout)
        
    def update_preview_on_effect_change(self):
        # Re-apply mask to update preview when effect changes
        self.apply_mask()

    def upload_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open image', '', 'Image files (*.jpg *.png *.bmp)')
        if fname:
            self.image = cv2.imread(fname)
            self.current_image = self.image.copy()
            self.mask_obj = Mask(self.current_image)
            self.show_image(self.current_image)

    def show_image(self, img, update_current=True):
        if update_current:
            self.current_image = img.copy()
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        self.img_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))

    def apply_mask(self):
        if self.current_image is None:
            return
        mask_type = self.mask_combo.currentText()
        effect_type = self.effect_combo.currentText()
        h, w = self.current_image.shape[:2]
        self.mask_type = mask_type
        # Set default mask position for draggable masks
        if mask_type == 'Full':
            self.mask_obj = Mask(self.current_image)
            self.mask_obj.create_full_mask()
            self.mask_pos = None
        elif mask_type == 'Rectangle':
            self.mask_pos = [w//4, h//4]
            self.mask_obj = Mask(self.current_image)
            self.mask_obj.create_rectangle_mask((self.mask_pos[0], self.mask_pos[1]), (self.mask_pos[0]+w//2, self.mask_pos[1]+h//2))
        elif mask_type == 'Circle':
            self.mask_pos = [w//2, h//2]
            self.mask_obj = Mask(self.current_image)
            self.mask_obj.create_circle_mask((self.mask_pos[0], self.mask_pos[1]), min(h, w)//4)
        elif mask_type == 'Polygon':
            pts = [(50,50), (w-50,50), (w//2,h-50)]
            self.mask_obj = Mask(self.current_image)
            self.mask_obj.create_polygon_mask(pts)
            self.mask_pos = None
        elif mask_type == 'Ellipse':
            self.mask_pos = [w//2, h//2]
            self.mask_obj = Mask(self.current_image)
            self.mask_obj.create_ellipse_mask((self.mask_pos[0], self.mask_pos[1]), (w//4, h//8), angle=30)
        elif mask_type == 'Checkerboard':
            self.mask_obj = Mask(self.current_image)
            self.mask_obj.create_checkerboard_mask(block_size=40)
            self.mask_pos = None
        elif mask_type == 'Stripe':
            self.mask_obj = Mask(self.current_image)
            self.mask_obj.create_stripe_mask(orientation='vertical', stripe_width=20, gap=20)
            self.mask_pos = None

        # Live preview: apply effect only to masked region
        preview_img = self.current_image.copy()
        mask = self.mask_obj.mask
        effect = None
        if effect_type == 'RollPixels':
            effect = RollPixelsEffect(self.current_image, which='row', shift_length=30, mask=mask)
        elif effect_type == 'RandomRollPixels':
            effect = RandomRollPixelsEffect(self.current_image, which='col', group_size=20, shift_range=(10, 40), mask=mask)
        elif effect_type == 'ColorValue':
            effect = ColorValueEffect(self.current_image, shift_value=80, mask=mask)
        elif effect_type == 'ColorChannelSplit':
            effect = ColorChannelSplitEffect(self.current_image, split_distance=40, which='row', order='bgr', mask=mask)
        elif effect_type == 'Corruption':
            effect = CorruptionEffect(self.current_image, corruption_type='random', bitsize=16, mask=mask)
        elif effect_type == 'Sort':
            effect = SortEffect(self.current_image, which='row', mask=mask)
        elif effect_type == 'PaletteReduction':
            effect = ColorPaletteReductionEffect(self.current_image, num_colors=8, mask=mask)
        if effect is not None:
            effected = effect.apply()
            # Only show effect in masked region, blend with original
            preview_img[mask == 255] = effected[mask == 255]
        # Draw dark outline around mask
        outline_img = preview_img.copy()
        outline_thickness = 3
        if mask_type == 'Rectangle' and self.mask_pos is not None:
            x, y = self.mask_pos
            pt1 = (int(x), int(y))
            pt2 = (int(x + w//2), int(y + h//2))
            cv2.rectangle(outline_img, pt1, pt2, (30,30,30), outline_thickness)
        elif mask_type == 'Circle' and self.mask_pos is not None:
            x, y = self.mask_pos
            radius = min(h, w)//4
            cv2.circle(outline_img, (int(x), int(y)), int(radius), (30,30,30), outline_thickness)
        elif mask_type == 'Ellipse' and self.mask_pos is not None:
            x, y = self.mask_pos
            axes = (w//4, h//8)
            cv2.ellipse(outline_img, (int(x), int(y)), axes, 30, 0, 360, (30,30,30), outline_thickness)
        self.show_image(outline_img, update_current=False)

    def mouse_press_event(self, event):
        if self.image is None or self.mask_pos is None:
            return
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.last_mouse_pos = event.pos()

    def mouse_move_event(self, event):
        if self.image is None or self.mask_pos is None or not self.dragging:
            return
        dx = event.x() - self.last_mouse_pos.x()
        dy = event.y() - self.last_mouse_pos.y()
        # Scale mouse movement to image coordinates
        pixmap = self.img_label.pixmap()
        if pixmap:
            img_w = pixmap.width()
            img_h = pixmap.height()
            scale_x = self.image.shape[1] / img_w
            scale_y = self.image.shape[0] / img_h
            dx = int(dx * scale_x)
            dy = int(dy * scale_y)
        self.mask_pos[0] = max(0, min(self.image.shape[1]-1, self.mask_pos[0]+dx))
        self.mask_pos[1] = max(0, min(self.image.shape[0]-1, self.mask_pos[1]+dy))
        self.last_mouse_pos = event.pos()
        # Redraw mask at new position
        self.update_mask_position()
    def mouse_release_event(self, event):
        self.dragging = False
        pass
        return
    def update_mask_position(self):
        h, w = self.current_image.shape[:2]
        if self.mask_type == 'Rectangle' and self.mask_pos is not None:
            self.mask_obj = Mask(self.current_image)
            x, y = self.mask_pos
            self.mask_obj.create_rectangle_mask((x, y), (x+w//2, y+h//2))
        elif self.mask_type == 'Circle' and self.mask_pos is not None:
            self.mask_obj = Mask(self.current_image)
            x, y = self.mask_pos
            self.mask_obj.create_circle_mask((x, y), min(h, w)//4)
        elif self.mask_type == 'Ellipse' and self.mask_pos is not None:
            self.mask_obj = Mask(self.current_image)
            x, y = self.mask_pos
            self.mask_obj.create_ellipse_mask((x, y), (w//4, h//8), angle=30)
        # Live preview: apply effect only to masked region
        effect_type = self.effect_combo.currentText()
        mask = self.mask_obj.mask
        preview_img = self.current_image.copy()
        effect = None
        if effect_type == 'RollPixels':
            effect = RollPixelsEffect(self.current_image, which='row', shift_length=30, mask=mask)
        elif effect_type == 'RandomRollPixels':
            effect = RandomRollPixelsEffect(self.current_image, which='col', group_size=20, shift_range=(10, 40), mask=mask)
        elif effect_type == 'ColorValue':
            effect = ColorValueEffect(self.current_image, shift_value=80, mask=mask)
        elif effect_type == 'ColorChannelSplit':
            effect = ColorChannelSplitEffect(self.current_image, split_distance=40, which='row', order='bgr', mask=mask)
        elif effect_type == 'Corruption':
            effect = CorruptionEffect(self.current_image, corruption_type='random', bitsize=16, mask=mask)
        elif effect_type == 'Sort':
            effect = SortEffect(self.current_image, which='row', mask=mask)
        elif effect_type == 'PaletteReduction':
            effect = ColorPaletteReductionEffect(self.current_image, num_colors=8, mask=mask)
        if effect is not None:
            effected = effect.apply()
            preview_img[mask == 255] = effected[mask == 255]
        # Draw dark outline around mask
        outline_img = preview_img.copy()
        outline_thickness = 3
        if self.mask_type == 'Rectangle' and self.mask_pos is not None:
            x, y = self.mask_pos
            pt1 = (int(x), int(y))
            pt2 = (int(x + w//2), int(y + h//2))
            cv2.rectangle(outline_img, pt1, pt2, (30,30,30), outline_thickness)
        elif self.mask_type == 'Circle' and self.mask_pos is not None:
            x, y = self.mask_pos
            radius = min(h, w)//4
            cv2.circle(outline_img, (int(x), int(y)), int(radius), (30,30,30), outline_thickness)
        elif self.mask_type == 'Ellipse' and self.mask_pos is not None:
            x, y = self.mask_pos
            axes = (w//4, h//8)
            cv2.ellipse(outline_img, (int(x), int(y)), axes, 30, 0, 360, (30,30,30), outline_thickness)
        self.show_image(outline_img, update_current=False)
            
    def apply_effect(self):
        effect_type = self.effect_combo.currentText()
        # Always apply effect to current_image, not original
        if effect_type == 'RollPixels':
            effect = RollPixelsEffect(self.current_image, which='row', shift_length=30, mask=self.mask_obj.mask)
        elif effect_type == 'RandomRollPixels':
            effect = RandomRollPixelsEffect(self.current_image, which='col', group_size=20, shift_range=(10, 40), mask=self.mask_obj.mask)
        elif effect_type == 'ColorValue':
            effect = ColorValueEffect(self.current_image, shift_value=80, mask=self.mask_obj.mask)
        elif effect_type == 'ColorChannelSplit':
            effect = ColorChannelSplitEffect(self.current_image, split_distance=40, which='row', order='bgr', mask=self.mask_obj.mask)
        elif effect_type == 'Corruption':
            effect = CorruptionEffect(self.current_image, corruption_type='random', bitsize=16, mask=self.mask_obj.mask)
        elif effect_type == 'Sort':
            effect = SortEffect(self.current_image, which='row', mask=self.mask_obj.mask)
        elif effect_type == 'PaletteReduction':
            effect = ColorPaletteReductionEffect(self.current_image, num_colors=8, mask=self.mask_obj.mask)
        out = effect.apply()
        self.show_image(out)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = ImageEffectGUI()
    gui.show()
    sys.exit(app.exec_())
