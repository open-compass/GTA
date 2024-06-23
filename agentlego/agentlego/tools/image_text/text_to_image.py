from agentlego.types import Annotated, ImageIO, Info
from agentlego.utils import require
from ..base import BaseTool
from ..utils.diffusers import load_sd, load_sdxl


class TextToImage(BaseTool):
    """A tool to generate image according to some keywords.

    Args:
        model (str): The stable diffusion model to use. You can choose
            from "sd" and "sdxl". Defaults to "sd".
        device (str): The device to load the model. Defaults to 'cuda'.
        toolmeta (None | dict | ToolMeta): The additional info of the tool.
            Defaults to None.
    """

    default_desc = ('This tool can generate an image according to the '
                    'input text.')

    @require('diffusers')
    def __init__(
        self,
        model: str = 'sdxl',
        num_inference_steps: int = 30,
        device: str = 'cuda',
        toolmeta=None,
    ):
        super().__init__(toolmeta=toolmeta)
        assert model in ['sd', 'sdxl', 'sdxl-turbo']
        self.model = model
        self.device = device
        self.num_inference_steps = num_inference_steps

    def setup(self):
        if self.model == 'sdxl':
            self.pipe = load_sdxl(device=self.device)
        elif self.model == 'sdxl-turbo':
            self.pipe = load_sdxl(model='stabilityai/sdxl-turbo', vae=None, device=self.device)
        elif self.model == 'sd':
            self.pipe = load_sd(device=self.device)
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, '\
                        ' missing fingers, extra digit, fewer digits, '\
                        'cropped, worst quality, low quality'

    def apply(
        self,
        keywords: Annotated[str,
                            Info('A series of keywords separated by comma.')],
    ) -> ImageIO:

        from urllib.parse import quote_plus

        import requests
        import langid

        langid.set_languages(['zh', 'en'])
        lang = langid.classify(keywords)[0]
        lang = 'zh-CN' if lang == 'zh' else lang
        if lang == 'zh-CN':
            keywords = quote_plus(keywords)
            url_tmpl = ('https://translate.googleapis.com/translate_a/'
                        'single?client=gtx&sl={}&tl={}&dt=at&dt=bd&dt=ex&'
                        'dt=ld&dt=md&dt=qca&dt=rw&dt=rm&dt=ss&dt=t&q={}')
            response = requests.get(url_tmpl.format('zh-CN', 'en', keywords), timeout=10).json()
            keywords = ''.join(x[0] for x in response[0] if x[0] is not None)
        prompt = f'{keywords}, {self.a_prompt}'
        image = self.pipe(
            prompt,
            num_inference_steps=self.num_inference_steps,
            negative_prompt=self.n_prompt,
        ).images[0]
        return ImageIO(image)
