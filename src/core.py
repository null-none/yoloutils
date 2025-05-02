import os
import numpy as np
import shapely.wkt


class YoloUtils:
    def __init__(self):
        self.image_width = 1280
        self.image_height = 720

    def non_negative(self, coord):
        """
        Sets negative coordinates to zero. This fixes bugs in some labeling tools.

        Input:
            coord: Int or float
            Any number that represents a coordinate, whether normalized or not.
        """

        if coord < 0:
            return 0
        else:
            return coord

    def polygon_to_bbox(self, polygon):
        """
        Converts a polygon to a bounding box.

        Args:
            polygon (list or np.ndarray): A list or array of (x, y) coordinates representing the polygon.

        Returns:
            tuple: Bounding box in the format (x_min, y_min, x_max, y_max).

        Example:
            polygon = [(1, 2), (4, 6), (7, 3), (5, 1)]
            bbox = polygon_to_bbox(polygon)
            print("Bounding Box:", bbox)
        """
        polygon = np.array(polygon)
        x_min = np.min(polygon[:, 0])
        y_min = np.min(polygon[:, 1])
        x_max = np.max(polygon[:, 0])
        y_max = np.max(polygon[:, 1])
        return x_min, y_min, x_max, y_max

    def polygon_to_yolo(self, polygon, img_width=100, img_height=100):
        """Convert string defining WKT polygon into YOLO format"""
        p = shapely.wkt.loads(polygon)
        x, y = p.centroid.x, p.centroid.y
        w, h = p.bounds[2] - p.bounds[0], p.bounds[3] - p.bounds[1]
        return x / img_width, y / img_height, w / img_width, h / img_height

    def pixel_to_yolo(self, dim, pixel_coords):
        """
        Transforms coordinates in YOLO format to coordinates in pixels.

        Input:
            dim: Tuple or list
            Image size (width, height).
            pixel_coords: List
            Bounding box coordinates in pixels (xmin, ymin, xmax, ymax).
        Output:
            yolo_coords: List
            Bounding box coordinates in YOLO format (xcenter, ycenter, width, height).
        """

        dw = 1 / dim[0]
        dh = 1 / dim[1]
        xcenter = self.non_negative(dw * (pixel_coords[0] + pixel_coords[2]) / 2)
        ycenter = self.non_negative(dh * (pixel_coords[1] + pixel_coords[3]) / 2)
        width = self.non_negative(dw * (pixel_coords[2] - pixel_coords[0]))
        height = self.non_negative(dh * (pixel_coords[3] - pixel_coords[1]))

        yolo_coords = [xcenter, ycenter, width, height]

        return yolo_coords

    def yolo_to_pixel(self, dim, yolo_coords):
        """
        Transforms coordinates in YOLO format to coordinates in pixels.

        Input:
            dim: Tuple or list
            Image size (width, height).
            yolo_coords: List
            Bounding box coordinates in YOLO format (xcenter, ycenter, width, height).
        Output:
            pixel_coords: List
            Bounding box coordinates in pixels (xmin, ymin, xmax, ymax).
        """

        xmin = self.non_negative(round(dim[0] * (yolo_coords[0] - yolo_coords[2] / 2)))
        xmax = self.non_negative(round(dim[0] * (yolo_coords[0] + yolo_coords[2] / 2)))
        ymin = self.non_negative(round(dim[1] * (yolo_coords[1] - yolo_coords[3] / 2)))
        ymax = self.non_negative(round(dim[1] * (yolo_coords[1] + yolo_coords[3] / 2)))

        pixel_coords = [xmin, ymin, xmax, ymax]

        return pixel_coords

    def seg_to_bbox(self, seg_info):
        # Example input: 5 0.046875 0.369141 0.0644531 0.384766 0.0800781 0.402344 ...
        class_id, *points = seg_info.split()
        points = [float(p) for p in points]
        x_min, y_min, x_max, y_max = (
            min(points[0::2]),
            min(points[1::2]),
            max(points[0::2]),
            max(points[1::2]),
        )
        width, height = x_max - x_min, y_max - y_min
        x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2
        bbox_info = f"{int(class_id)-1} {x_center} {y_center} {width} {height}"
        return bbox_info

    def center_size_to_bbox(self, center, size):
        """
        Convert a rectangle defined by center (x, y) and size (width, height) to a bounding box.

        Args:
            center (tuple): (x, y) coordinates of the rectangle center.
            size (tuple): (width, height) of the rectangle.

        Returns:
            tuple: (xmin, ymin, xmax, ymax) bounding box.
        """
        cx, cy = center
        width, height = size

        half_width = width / 2
        half_height = height / 2

        xmin = cx - half_width
        ymin = cy - half_height
        xmax = cx + half_width
        ymax = cy + half_height

        return xmin, ymin, xmax, ymax

    def annotation_to_cv2(self, input_file, output_file):
        with open(input_file, "r") as f:
            lines = f.readlines()

        new_lines = list()
        for line in lines:
            data = line.strip().split(" ")

            class_id = int(data[0])
            x_center = float(data[1])
            y_center = float(data[2])
            width = float(data[3])
            height = float(data[4])

            x_min = int((x_center - (width / 2)) * self.image_width)
            y_min = int((y_center - (height / 2)) * self.image_height)
            x_max = int((x_center + (width / 2)) * self.image_width)
            y_max = int((y_center + (height / 2)) * self.image_height)

            new_data = f"{class_id} {x_min} {y_min} {x_max} {y_max}\n"
            new_lines.append(new_data)

        with open(output_file, "w") as f:
            f.writelines(new_lines)
