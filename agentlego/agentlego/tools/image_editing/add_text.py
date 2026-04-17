
from typing import Optional
from PIL import ImageDraw, ImageFont

from agentlego.types import Annotated, ImageIO, Info
from agentlego.utils import parse_multi_float
from ..base import BaseTool


class AddText(BaseTool):
    default_desc = 'A tool to draw a box on a certain region of the input image.'

    def apply(
        self,
        image: ImageIO,
        text: str,
        position: Annotated[
            str,
            Info('The left-bottom corner coordinate in the format of `(x, y)`, '
                 'or a combination of ["l"(left), "m"(middle), "r"(right)] '
                 'and ["t"(top), "m"(middle), "b"(bottom)] like "mt" for middle-top')],
        color: str = 'red',
        output_dir: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> ImageIO:
        image_pil = image.to_pil().copy()
        draw = ImageDraw.Draw(image_pil)

        # Estimate a proper font size.
        fontsize = int(image_pil.size[1] / 336 * 18)
        font = ImageFont.load_default(size=fontsize)
        w, h = image_pil.size
        m = w//20  # margin
        POS = {
            'lt': (m, m),
            'lm': (m, h // 2),
            'lb': (m, h - m),
            'mt': (w // 2, m),
            'mm': (w // 2, h // 2),
            'mb': (w // 2, h - m),
            'rt': (w - m, m),
            'rm': (w - m, h // 2),
            'rb': (w - m, h - m),
        }
        if position in POS:
            xy = POS[position]
            anchor = position
        else:
            anchor = 'lb'
            try:
                x, y = parse_multi_float(position, 2)
                xy = (x, y)
            except ValueError:
                raise ValueError('Invalid position string.')
        draw.text(xy, text, anchor=anchor, fill=color, font=font)
        import os
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            if filename is None:
                filename = 'image_with_text.png'
            output_path = os.path.join(output_dir, filename)
            image_pil.save(output_path)
            return ImageIO(output_path)
        else:
            return ImageIO(image_pil)
