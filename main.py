import gradio as gr
import numpy as np
from PIL import Image

from neural_network import NeuralNetwork


def grayscale(arr):
    height, width = arr.shape[0:2]
    grayscaled = np.empty((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            grayscaled[y][x] = arr[y][x][3]
    return grayscaled


def minimum_bounding_box(arr):
    height, width = arr.shape
    left = width
    top = height
    right = 0
    bottom = 0
    for y in range(height):
        for x in range(width):
            if arr[y][x] == 0:
                continue
            if x < left:
                left = x
            if x > right:
                right = x
            if y < top:
                top = y
            if y > bottom:
                bottom = y
    return (left, top, right, bottom)


def trim(image, box):
    return image.crop(box)


def normalize(image, size):
    width, height = image.size
    max_width, max_height = size
    scale = min(max_width / width, max_height / height)
    return image.resize((int(width * scale), int(height * scale)))


def center_of_mass(arr):
    height, width = arr.shape
    R = np.zeros(2)
    for y in range(height):
        for x in range(width):
            m = arr[y][x]
            r = np.array([x, y])
            R += m * r
    R /= np.sum(arr)
    return R


def center(image, size):
    new_width, new_height = size
    center_of_new_image = np.array([new_width / 2, new_height / 2])
    R = center_of_mass(np.array(image))
    offset = center_of_new_image - R
    new_image = Image.new("L", size)
    new_image.paste(image, tuple(offset.astype(int)))
    return new_image


def predict(im):
    # The original black and white (bilevel) images from NIST were size normalized to fit in a 20x20 pixel box while preserving their aspect ratio. The resulting images contain grey levels as a result of the anti-aliasing technique used by the normalization algorithm. the images were centered in a 28x28 image by computing the center of mass of the pixels, and translating the image so as to position this point at the center of the 28x28 field.
    grayscaled = grayscale(im["composite"])
    bbox = minimum_bounding_box(grayscaled)
    trimmed = trim(Image.fromarray(grayscaled), bbox)
    normalized = normalize(trimmed, (20, 20))
    centered = np.array(center(normalized, (28, 28)))

    input = np.empty((28 * 28, 1))
    for y in range(28):
        for x in range(28):
            input[y * 28 + x] = [centered[y][x] / 255]
    output = neural_network.feedforward(input)
    return int(np.argmax(output))


if __name__ == "__main__":
    neural_network = NeuralNetwork.deserialize("model.json")

    demo = gr.Interface(
        fn=predict,
        inputs=[gr.ImageEditor(sources=(), crop_size=(100, 100), transforms=(), layers=False)],
        outputs=["label"],
    )
    demo.launch()
