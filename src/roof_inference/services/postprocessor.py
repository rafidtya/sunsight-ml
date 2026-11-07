import uuid
import numpy as np

def calculate_polygon_area(polygon):
    """Calculate area using shoelace formula."""
    if not polygon or len(polygon) < 3:
        return 0.0
    x = [p[0] for p in polygon]
    y = [p[1] for p in polygon]
    n = len(x)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += x[i] * y[j]
        area -= x[j] * y[i]
    return abs(area) / 2.0

def process_results(result, min_confidence=0.25):
    detections = []
    if result.masks is None or result.boxes is None:
        return detections
    
    polygons = result.masks.xy

    for index, polygon in enumerate(polygons):
        box = result.boxes[index]
        confidence = float(box.conf.item())
        
        if confidence < min_confidence:
            continue

        class_id = int(box.cls.item())
        bbox = box.xyxy[0].tolist()
        polygon_points = [[float(x), float(y)] for x, y in polygon]
        area_px = calculate_polygon_area(polygon_points)

        detections.append({
            "id": f"det_{uuid.uuid4().hex[:8]}",
            "class_id": class_id,
            "class_name": result.names[class_id],
            "confidence": confidence,
            "bbox": {
                "x1": round(bbox[0], 2),
                "y1": round(bbox[1], 2),
                "x2": round(bbox[2], 2),
                "y2": round(bbox[3], 2),
            },
            "polygon": polygon_points,
            "area_px": round(area_px, 2),
        })
    
    return detections