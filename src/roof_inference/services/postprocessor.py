def process_results(result):
    detections = []
    if result.masks is None or result.boxes is None:
        return detections
    
    polygons = result.masks.xy

    for index, polygon in enumerate(polygons):
        box = result.boxes[index]

        class_id = int(box.cls.item())
        bbox = box.xyxy[0].tolist()


        detections.append({
            "class_id": class_id,
            "class_name": result.names[class_id],
            "confidence": float(box.conf.item()),
            "bbox":{
                "x1": round(bbox[0],2),
                "y1": round(bbox[1],2),
                "x2": round(bbox[2],2),
                "y2": round(bbox[3],2),
            },
            "polygon": [
                [round(float(x),2), round(float(y),2)]
                for x,y in polygon
            ],
        })
    
    return detections