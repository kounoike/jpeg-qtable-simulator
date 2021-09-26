import pyvirtualcam
import click
from PIL import Image
from io import BytesIO
import numpy as np

def process_jpeg_frame(frame, qtables):
    with BytesIO(frame) as f:
        with Image.open(f) as jpeg:
            img = jpeg.copy()
    
    with BytesIO() as f:
        img.save(f, "jpeg", qtables=qtables)
        f.seek(0)
        with Image.open(f) as decoded:
            return decoded.copy()


@click.command()
@click.argument('mjpeg_path')
@click.argument('jpeg_path')
def run(mjpeg_path, jpeg_path):
    with Image.open(jpeg_path) as jpeg:
        qtables = jpeg.quantization

    with open(mjpeg_path, 'rb') as mjpeg_file:
        mjpeg_data = mjpeg_file.read()
    soi = b'\xff\xd8'
    eoi = b'\xff\xd9'
    soi_idx = mjpeg_data.find(soi)

    image_list = []

    with click.progressbar(length=len(mjpeg_data), label='reading mjpeg data...') as bar:
        while soi_idx > -1:
            eoi_idx = mjpeg_data.find(eoi, soi_idx)
            bar.update(eoi_idx - soi_idx)
            ret = process_jpeg_frame(mjpeg_data[soi_idx:eoi_idx + len(eoi)], qtables)
            image_list.append(ret)
            soi_idx = mjpeg_data.find(soi, eoi_idx)
            break

    click.echo('done.')

    width = image_list[0].width
    height = image_list[0].height

    print(width, height)

    with pyvirtualcam.Camera(width=width, height=height, fps=30) as cam:
        frame_idx = 0
        while True:
            frame = image_list[frame_idx % len(image_list)]
            cam.send(np.array(frame), dtype=np.uint8)
            cam.sleep_until_next_frame()


if __name__ == '__main__':
    run()
