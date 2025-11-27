# %% [markdown]
# # Objects Locallization
# ## Table of contents
# - [1 - Import Library](#section1)
# - [2 - Section 2](#section2)
# - [3 - Section 3](#section3)
# %%
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import load_model
import yad2k.utils.utils as utils
import yad2k.models.keras_yolo as keras_yolo
import os

# %%
yolo_model = load_model("model_data", compile=False)
yolo_model.summary()
# %%

anchors = utils.read_anchors("model_data/yolo_anchors.txt")
class_names = utils.read_classes("model_data/coco_classes.txt")


# %%
def yolo_boxes_to_corners(box_xy, box_wh):
    box_mins = box_xy - (box_wh / 2.0)
    box_maxes = box_xy + (box_wh / 2.0)

    return tf.keras.backend.concatenate(
        [
            box_mins[..., 1:2],  # y_min
            box_mins[..., 0:1],  # x_min
            box_maxes[..., 1:2],  # y_max
            box_maxes[..., 0:1],  # x_max
        ]
    )


# %%


def yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold=0.6):
    box_scores = box_confidence * box_class_probs

    box_classes = tf.math.argmax(box_scores, axis=-1)  # (19, 19, 5)
    box_class_scores = tf.math.reduce_max(box_scores, axis=-1)  # (19, 19, 5)

    filtering_mask = box_class_scores >= threshold  # (19, 19, 5)

    scores = tf.boolean_mask(box_class_scores, filtering_mask)  # (19, 19, ???)
    boxes = tf.boolean_mask(boxes, filtering_mask)  # (19, 19, ???, 4)
    classes = tf.boolean_mask(box_classes, filtering_mask)  # (19, 19, ???)

    return scores, boxes, classes


# %%


def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):
    boxes = tf.cast(boxes, dtype=tf.float32)
    scores = tf.cast(scores, dtype=tf.float32)

    nms_indices = []
    classes_labels = tf.unique(classes)[0]  # Get unique classes

    for label in classes_labels:
        filtering_mask = classes == label

        boxes_label = tf.boolean_mask(boxes, filtering_mask)

        scores_label = tf.boolean_mask(scores, filtering_mask)

        if tf.shape(scores_label)[0] > 0:  # Check if there are any boxes to process
            nms_indices_label = tf.image.non_max_suppression(
                boxes_label, scores_label, max_boxes, iou_threshold=iou_threshold
            )

            selected_indices = tf.squeeze(tf.where(filtering_mask), axis=1)

            nms_indices.append(tf.gather(selected_indices, nms_indices_label))

    nms_indices = tf.concat(nms_indices, axis=0)

    scores = tf.gather(scores, nms_indices)
    boxes = tf.gather(boxes, nms_indices)
    classes = tf.gather(classes, nms_indices)

    sort_order = tf.argsort(scores, direction="DESCENDING").numpy()
    scores = tf.gather(scores, sort_order[0:max_boxes])
    boxes = tf.gather(boxes, sort_order[0:max_boxes])
    classes = tf.gather(classes, sort_order[0:max_boxes])

    return scores, boxes, classes


# %%


def yolo_eval(
    yolo_outputs,
    image_shape=(720, 1280),
    max_boxes=10,
    score_threshold=0.6,
    iou_threshold=0.5,
):
    box_xy, box_wh, box_confidence, box_class_probs = yolo_outputs

    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    scores, boxes, classes = yolo_filter_boxes(
        boxes,  # Use boxes
        box_confidence,  # Use box confidence
        box_class_probs,  # Use box class probability
        score_threshold,  # Use threshold=score_threshold
    )

    boxes = utils.scale_boxes(boxes, image_shape)
    scores, boxes, classes = yolo_non_max_suppression(
        scores,  # Use scores
        boxes,  # Use boxes
        classes,  # Use classes
        max_boxes,  # Use max boxes
        iou_threshold,  # Use iou_threshold=iou_threshold
    )

    return scores, boxes, classes


# %%
def predict(image_file):
    image, image_data = utils.preprocess_image(
        "images/" + image_file, model_image_size=(608, 608)
    )

    yolo_model_outputs = yolo_model(image_data)
    yolo_outputs = keras_yolo.yolo_head(yolo_model_outputs, anchors, len(class_names))

    out_scores, out_boxes, out_classes = yolo_eval(
        yolo_outputs, [image.size[1], image.size[0]], 10, 0.3, 0.5
    )

    print("Found {} boxes for {}".format(len(out_boxes), "images/" + image_file))
    colors = utils.get_colors_for_classes(len(class_names))
    utils.draw_boxes(image, out_boxes, out_classes, class_names, out_scores)
    image.save(os.path.join("out", image_file), quality=100)
    output_image = Image.open(os.path.join("out", image_file))
    plt.imshow(output_image)

    return out_scores, out_boxes, out_classes


# %%
print(predict("test.jpg"))
