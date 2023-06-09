import cv2
import json
import os
import torch
import torchvision
import torch.nn as nn
import numpy as np
import warnings
import logging
from PIL import ImageFont, ImageDraw, Image
from tqdm import tqdm
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

logger = logging.getLogger("detectron2")
logger.setLevel(logging.CRITICAL)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_IMAGES_PATH = 'img'
SAVE_PATH = 'res'
SEGM_MODEL_PATH = "Models/model_0059999.pth"
OCR_MODEL_PATH = "Models/baseline_model_self_trained.ckpt"
# OCR_MODEL_PATH = "Models/cool_model.ckpt"

config_json = {
    # "alphabet": """@ !"%'()+,-./0123456789:;=?EFIMNOSTW[]abcdefghiklmnopqrstuvwxyАБВГДЕЖЗИКЛМНОПРСТУХЦЧШЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяё№""",
    "alphabet": r'!"%\'()*+,-./0123456789:;<=>?ABCDEFGHIJKLMNOPRSTUVWXY['
                r']_abcdefghijklmnopqrstuvwxyz|}ЁАБВГДЕЖЗИКЛМНОПРСТУФХЦЧШЩЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяё№',
    "image": {
        "width": 256,
        "height": 32
    }
}


def get_contours_from_mask(mask, min_area=5):
    contours, hierarchy = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    contour_list = []
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            contour_list.append(contour)
    return contour_list


def get_larger_contour(contours):
    larger_area = 0
    larger_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > larger_area:
            larger_contour = contour
            larger_area = area
    return larger_contour


def black2white(image):
    lo = np.array([0, 0, 0])
    hi = np.array([0, 0, 0])
    mask = cv2.inRange(image, lo, hi)
    image[mask > 0] = (255, 255, 255)
    return image


class SEGMpredictor:
    def __init__(self, model_path):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))

        cfg.MODEL.WEIGHTS = model_path
        cfg.TEST.EVAL_PERIOD = 1000

        cfg.INPUT.MIN_SIZE_TRAIN = 2160
        cfg.INPUT.MAX_SIZE_TRAIN = 3130

        cfg.INPUT.MIN_SIZE_TEST = 2160
        cfg.INPUT.MAX_SIZE_TEST = 3130
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
        cfg.INPUT.FORMAT = 'BGR'
        cfg.DATALOADER.NUM_WORKERS = 4
        cfg.SOLVER.IMS_PER_BATCH = 3
        cfg.SOLVER.BASE_LR = 0.01
        cfg.SOLVER.GAMMA = 0.1
        cfg.SOLVER.STEPS = (1500,)

        cfg.SOLVER.MAX_ITER = 17000
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.SOLVER.CHECKPOINT_PERIOD = cfg.TEST.EVAL_PERIOD
        cfg.TEST.DETECTIONS_PER_IMAGE = 1000
        cfg.OUTPUT_DIR = './output'

        self.predictor = DefaultPredictor(cfg)

    def __call__(self, img):
        outputs = self.predictor(img)
        prediction = outputs["instances"].pred_masks.cpu().numpy()
        contours = []
        for pred in prediction:
            contour_list = get_contours_from_mask(pred)
            contours.append(get_larger_contour(contour_list))
        return contours


OOV_TOKEN = '<OOV>'
CTC_BLANK = '<BLANK>'


def get_char_map(alphabet):
    """Make from string alphabet character to int dict.
    Add BLANK char from CTC loss and OOV char for out of vocabulary symbols."""

    char_map = {value: idx + 2 for (idx, value) in enumerate(alphabet)}
    char_map[CTC_BLANK] = 0
    char_map[OOV_TOKEN] = 1
    return char_map


class Tokenizer:
    """Class for encoding and decoding string word to sequence of int
    (and vice versa) using alphabet."""

    def __init__(self, alphabet):
        self.char_map = get_char_map(alphabet)
        self.rev_char_map = {val: key for key, val in self.char_map.items()}

    def encode(self, word_list):
        """Returns a list of encoded words (int)."""

        enc_words = []
        for word in word_list:
            enc_words.append(
                [self.char_map[char] if char in self.char_map
                 else self.char_map[OOV_TOKEN]
                 for char in word]
            )
        return enc_words

    def get_num_chars(self):
        return len(self.char_map)

    def decode(self, enc_word_list):
        """Returns a list of words (str) after removing blanks and collapsing
        repeating characters. Also skip out of vocabulary token."""

        dec_words = []
        for word in enc_word_list:
            word_chars = ''
            for idx, char_enc in enumerate(word):
                # skip if blank symbol, oov token or repeated characters
                if (
                        char_enc != self.char_map[OOV_TOKEN]
                        and char_enc != self.char_map[CTC_BLANK]
                        # idx > 0 to avoid selecting [-1] item
                        and not (idx > 0 and char_enc == word[idx - 1])
                ):
                    word_chars += self.rev_char_map[char_enc]
            dec_words.append(word_chars)
        return dec_words


class ImageResize:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, image):
        image = cv2.resize(image, (self.width, self.height),
                           interpolation=cv2.INTER_LINEAR)
        return image


class MoveChannels:
    def __call__(self, image):
        return np.moveaxis(image, -1, 0)


class Normalize:
    def __call__(self, img):
        return img.astype(np.float32) / 255


class ToTensor:
    def __call__(self, arr):
        return torch.from_numpy(arr)


def get_resnet34_backbone():
    m = torchvision.models.resnet34(pretrained=True)
    input_conv = nn.Conv2d(3, 64, 7, 1, 3)
    blocks = [input_conv, m.bn1, m.relu,
              m.maxpool, m.layer1, m.layer2, m.layer3]
    return nn.Sequential(*blocks)


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return out


class RCNN(nn.Module):
    # def __init__(
    #         self, number_class_symbols, out_len=32
    # ):
    #     super().__init__()
    #     self.feature_extractor = get_resnet34_backbone()
    #     self.avg_pool = nn.AdaptiveAvgPool2d(
    #         (512, out_len))
    #     self.bilstm = BiLSTM(512, 256, 2)
    #     self.classifier = nn.Sequential(
    #         nn.Linear(512, 256),
    #         nn.GELU(),
    #         nn.Dropout(0.1),
    #         nn.Linear(256, number_class_symbols)
    #     )
    #
    # def forward(self, x, return_x=False):
    #     feature = self.feature_extractor(x)
    #     b, c, h, w = feature.size()
    #     feature = feature.view(b, c * h, w)
    #     feature = self.avg_pool(feature)
    #     feature = feature.transpose(1, 2)
    #     out = self.bilstm(feature)
    #     # print(x.shape)
    #     out = self.classifier(out)
    #     x1 = nn.functional.log_softmax(out, dim=2).permute(1, 0, 2)
    #     if return_x:
    #         return x1, out
    #     else:
    #         return x1

    def __init__(
            self, number_class_symbols, time_feature_count=256, lstm_hidden=256,
            lstm_len=2,
    ):
        super().__init__()
        self.feature_extractor = get_resnet34_backbone()
        self.avg_pool = nn.AdaptiveAvgPool2d(
            (time_feature_count, time_feature_count))
        self.bilstm = BiLSTM(time_feature_count, lstm_hidden, lstm_len)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, time_feature_count),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(time_feature_count, number_class_symbols)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        b, c, h, w = x.size()
        x = x.view(b, c * h, w)
        x = self.avg_pool(x)
        x = x.transpose(1, 2)
        x = self.bilstm(x)
        x = self.classifier(x)
        x = nn.functional.log_softmax(x, dim=2).permute(1, 0, 2)
        return x


def predict(images, model, tokenizer, device):
    model.eval()
    images = images.to(device)
    with torch.no_grad():
        output = model(images)
    pred = torch.argmax(output.detach().cpu(), -1).permute(1, 0).numpy()
    text_predictions = tokenizer.decode(pred)
    return text_predictions


class InferenceTransform:
    def __init__(self, height, width):
        self.transforms = torchvision.transforms.Compose([
            ImageResize(height, width),
            MoveChannels(),
            Normalize(),
            ToTensor()
        ])

    def __call__(self, images):
        transformed_images = []
        for image in images:
            image = self.transforms(image)
            transformed_images.append(image)
        transformed_tensor = torch.stack(transformed_images, 0)
        return transformed_tensor


class OcrPredictor:
    def __init__(self, model_path, config, device=DEVICE):
        self.tokenizer = Tokenizer(config['alphabet'])
        self.device = torch.device(device)
        self.model = RCNN(number_class_symbols=self.tokenizer.get_num_chars())
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)

        self.transforms = InferenceTransform(
            height=config['image']['height'],
            width=config['image']['width'],
        )

    def __call__(self, images):
        if isinstance(images, (list, tuple)):
            one_image = False
        elif isinstance(images, np.ndarray):
            images = [images]
            one_image = True
        else:
            raise Exception(f"Input must contain np.ndarray, "
                            f"tuple or list, found {type(images)}.")

        images = self.transforms(images)
        pred = predict(images, self.model, self.tokenizer, self.device)

        if one_image:
            return pred[0]
        else:
            return pred


def crop_img_by_polygon(img, polygon):
    # https://stackoverflow.com/questions/48301186/cropping-concave-polygon-from-image-using-opencv-python
    pts = np.array(polygon)
    rect = cv2.boundingRect(pts)
    x, y, w, h = rect
    cropped = img[y:y + h, x:x + w].copy()
    pts = pts - pts.min(axis=0)
    mask = np.zeros(cropped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    dst = cv2.bitwise_and(cropped, cropped, mask=mask)
    return dst


class PipelinePredictor:
    def __init__(self, segm_model_path, ocr_model_path, ocr_config):
        self.segm_predictor = SEGMpredictor(model_path=segm_model_path)
        self.ocr_predictor = OcrPredictor(
            model_path=ocr_model_path,
            config=ocr_config
        )

    def __call__(self, img):
        output = {'predictions': []}
        contours = self.segm_predictor(img)
        for contour in contours:
            if contour is not None:
                crop = crop_img_by_polygon(img, contour)
                pred_text = self.ocr_predictor(crop)
                output['predictions'].append(
                    {
                        'polygon': [[int(i[0][0]), int(i[0][1])] for i in contour],
                        'text': pred_text
                    }
                )
        return output


def get_pipeline_predictor():
    return PipelinePredictor(
        segm_model_path=SEGM_MODEL_PATH,
        ocr_model_path=OCR_MODEL_PATH,
        ocr_config=config_json,
    )


def add_border(img, size):
    height, width, _ = img.shape
    top_border_size = int(height * size)
    side_border_size = int(((top_border_size * 2 + height) / 9 * 16 - width) / 2)
    white = [255, 255, 255]
    border = cv2.copyMakeBorder(
        img,
        top=top_border_size,
        bottom=top_border_size,
        left=side_border_size,
        right=side_border_size,
        borderType=cv2.BORDER_CONSTANT,
        value=white
    )
    border = cv2.copyMakeBorder(
        border,
        top=10,
        bottom=10,
        left=10,
        right=10,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )
    return border


def visualise_recognition(img, pred_data, font_path, font_coefficient=50, draw_type="contours"):
    """Draw concatenation of original image with drawn contours/rectangles and recognised words."""

    h, w = img.shape[:2]
    font = ImageFont.truetype(font_path, int(h / font_coefficient))
    empty_img = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(empty_img)

    for prediction in pred_data['predictions']:
        polygon = prediction['polygon']
        pred_text = prediction['text']

        if draw_type == "contours":
            cv2.drawContours(img, np.array([polygon]), -1, (150, 250, 0), 2)
        else:
            cv2.rectangle(img, cv2.boundingRect(np.array([polygon])), (150, 250, 0), 2)

        x, y, w, h = cv2.boundingRect(np.array([polygon]))
        cv2.circle(img, (x, y), 4, (0, 0, 250), -1)
        cv2.circle(img, (int(x + w / 2), int(y + h / 2)), 4, (250, 0, 200), -1)
        draw.text((x, y), pred_text, fill=0, font=font)

    vis_img = np.array(empty_img)
    vis = np.concatenate((img, vis_img), axis=1)
    return vis


class Word:
    def __init__(self, x, y, w, h, text):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.text = text
        self.center_x = int(x + w / 2)
        self.center_y = int(y + h / 2)
        self.area = w * h

    def __repr__(self):
        return f"({self.x}, {self.y}, {self.w}, {self.h}, c_x={self.center_x}, c_y={self.center_y}, {self.text})"

    def __str__(self):
        return f"(x={self.x}, y={self.y}, w={self.w}, h={self.h}, area={self.area}, {self.text})"


def intersection_area(a, b):
    """Finds intersection area of two given words."""
    
    x_min1 = a.x
    y_min1 = a.y
    x_max1 = a.x + a.w
    y_max1 = a.y + a.h
    x_min2 = b.x
    y_min2 = b.y
    x_max2 = b.x + b.w
    y_max2 = b.y + b.h
    left = max(x_min1, x_min2)
    bottom = max(y_min1, y_min2)
    right = min(x_max1, x_max2)
    top = min(y_max1, y_max2)
    width = right - left
    height = top - bottom
    if width <= 0 or height <= 0:
        return 0
    return width * height


def recognise(read_path, save_path=SAVE_PATH, output_type="easy", draw_type="contours"):
    """output_type = "easy" or "full". Returns either only bounding boxes or complete contours.\n
        draw_type = "contours" if on an output image you want exact contours to be drawn, or "rect" if you want only
        bounding boxes.
    """

    pipeline_predictor = get_pipeline_predictor()
    image = cv2.imread(read_path)
    image = add_border(image, size=0.6)
    prediction = pipeline_predictor(image)
    vis = visualise_recognition(image, prediction, 'font.otf', 50, draw_type)

    red = vis[:, :, 2].copy()
    blue = vis[:, :, 0].copy()
    vis[:, :, 0] = red
    vis[:, :, 2] = blue

    img = Image.fromarray(vis, 'RGB')
    img.save(os.path.join(save_path, os.path.basename(read_path)))

    if output_type == "easy":
        easier_output = {'predictions': []}
        for word in prediction["predictions"]:
            x, y, w, h = cv2.boundingRect(np.array(word["polygon"]))
            easier_output["predictions"].append(Word(x, y, w, h, word["text"]))
        return easier_output

    else:
        return prediction


def convert_to_text(prediction):
    """Converts raw prediction to text with line handling."""

    if not prediction['predictions']:
        return "<b><i>Текст на картинке не найден.</i></b>"
    sorted_words = sorted(prediction["predictions"], key=lambda w: w.center_y)
    lines = [[]]
    mean_y = sorted_words[0].center_y
    for word in sorted_words:
        if word.center_y > int(mean_y + word.h / 1.8):
            lines.append([])
            mean_y = word.center_y
        else:
            mean_y = int((mean_y + word.center_y) / 2)
        lines[len(lines) - 1].append(word)

    sorted_lines = [sorted(line, key=lambda w: w.x) for line in lines]
    ans = ''
    for line in sorted_lines:
        stack = []
        if len(line) == 1:
            if line[0].text != "." and line[0].text != "," and line[0].text != "-":
                ans += (line[0].text + '\n')
            continue
        stack.append(line[0])
        for word in line:
            if intersection_area(stack[-1], word) > 0.5 * min(stack[-1].area, word.area):
                if stack[-1].area < word.area:
                    stack[-1] = word
                else:
                    continue
            else:
                stack.append(word)
        for word in stack:
            ans += word.text + " "

        ans += "\n"

    ans = ans.replace(' ,', ',').replace(' .', '.').replace(' )', ')').replace('\n.', '\n').replace('\n,', '\n')
    ans = ans.replace('\n ', '\n').replace(',', ', ').replace('  ', ' ')
    ans = ans.replace('\n.', '\n').replace('\n,', '\n').replace('\n\n', '\n').replace('\n \n', '\n')
    ans = ans[:-1]
    return ans


def main():
    for img_name in tqdm(os.listdir(TEST_IMAGES_PATH)):
        pred_data = recognise(read_path=os.path.join(TEST_IMAGES_PATH, img_name),
                              save_path=SAVE_PATH,
                              output_type="easy",
                              draw_type="rect"
                              )
        print("Image " + img_name + ':\n"' + convert_to_text(pred_data) + '"')


if __name__ == '__main__':
    main()
    
