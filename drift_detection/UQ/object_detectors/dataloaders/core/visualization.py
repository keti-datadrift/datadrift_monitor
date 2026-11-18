import os

import cv2
import matplotlib.pyplot as plt


def plot_image_with_boxes(img, boxes, pred_cls, confidence, output_path, image_id, save=True):
    text_size = 0.6
    text_th = 2
    rect_th = 3
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255),
        (255, 255, 0), (255, 165, 0), (128, 0, 128), (255, 105, 180), (0, 255, 0),
    ]

    for i in range(len(boxes)):
        x1, y1, x2, y2 = [int(v) for v in boxes[i]]
        color = colors[i % len(colors)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color=color, thickness=rect_th)
        score = float(confidence[i]) if len(confidence) > i else 0.0
        label = pred_cls[i] if len(pred_cls) > i else "obj"
        cv2.putText(
            img,
            f"{label} {score:.2f}",
            (x1 + 5, y1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_size,
            (0, 150, 255),
            thickness=text_th,
        )

    if not save:
        return img

    os.makedirs(output_path, exist_ok=True)
    file_stem = os.path.splitext(os.path.basename(str(image_id)))[0]
    fig = plt.figure(figsize=(10, 7))
    plt.axis("off")
    plt.imshow(img)
    plt.savefig(os.path.join(output_path, f"{file_stem}.jpg"), dpi=fig.dpi)
    plt.close(fig)
