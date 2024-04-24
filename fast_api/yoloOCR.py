import os
from ultralytics import YOLO
import numpy as np
from pytesseract import image_to_string


model_path = "model"
general_model_name = "e50_aug.pt"
image_model_name = "e100_img.pt"

general_model = YOLO(os.path.join(model_path, general_model_name))
image_model = YOLO(os.path.join(model_path, image_model_name))


flags = {"hist": False, "bz": False}


configs = {}
configs["paratext"] = {"sz": 640, "conf": 0.25, "rm": True, "classes": [0, 1]}
configs["imgtab"] = {"sz": 640, "conf": 0.35, "rm": True, "classes": [2, 3]}
configs["image"] = {"sz": 640, "conf": 0.35, "rm": True, "classes": [0]}


def get_predictions(model, img2, config):
    res_dict = {"status": 1}
    try:
        for result in model.predict(
            source=img2,
            verbose=False,
            retina_masks=config["rm"],
            imgsz=config["sz"],
            conf=config["conf"],
            stream=True,
            classes=config["classes"],
            agnostic_nms=True,
        ):
            try:
                res_dict["masks"] = result.masks.data
                res_dict["boxes"] = result.boxes.data
                res_dict["xyxy"] = result.boxes.xyxy.cpu()

                del result
                return res_dict
            except Exception as e:
                res_dict["status"] = 0
                return res_dict
    except:
        res_dict["status"] = -1
        return res_dict


def get_masks(img, model, img_model, flags, configs):
    response = {"status": 1}
    ans_masks = []
    img2 = img

    res = get_predictions(model, img2, configs["paratext"])
    if res["status"] == -1:
        response["status"] = -1
        return response
    response["boxes1"] = np.array(res["xyxy"], dtype=np.int32)
    return response


def yoloTesseract(
    img, model=general_model, img_model=image_model, configs=configs, flags=flags
):

    # img = cv2.imread(img_path)
    res = get_masks(img, general_model, image_model, flags, configs)
    if res["status"] == -1:
        for idx in configs.keys():
            configs[idx]["rm"] = False
        return yoloTesseract(img, model, img_model, flags, configs)

    sorted_arr = res["boxes1"][res["boxes1"][:, 1].argsort()]
    text = ""
    for cords in sorted_arr:
        text += image_to_string(
            img[cords[1] : cords[3], cords[0] : cords[2]], lang="mal+eng"
        )
    # print(text)
    return text


# output = evaluate(
#     img_path="YOLOX/test.jpeg",
#     model=general_model,
#     img_model=image_model,
#     configs=configs,
#     flags=flags,
# )

# print(output)
