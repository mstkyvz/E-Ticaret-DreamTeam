import json
from pathlib import Path

class Config:
    def __init__(self, config_path='config/config.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

    def get(self, *keys):
        value = self.config
        for key in keys:
            value = value[key]
        return value

    @property
    def model_ckpt(self):
        return self.get('model', 'ckpt')

    @property
    def api_url(self):
        return self.get('api', 'url')

    @property
    def supported_languages(self):
        return self.get('languages', 'supported')

    @property
    def language_map(self):
        return self.get('languages', 'map')

    @property
    def image_max_dimensions(self):
        return self.get('image', 'max_width'), self.get('image', 'max_height')

    @property
    def flags_path(self):
        return Path(self.get('paths', 'flags'))

    @property
    def empty_image_path(self):
        return Path(self.get('paths', 'empty_image'))

    @property
    def output_image_path(self):
        return Path(self.get('paths', 'output_image'))

    @property
    def language_prompts_path(self):
        return Path(self.get('paths', 'language_prompts'))

    @property
    def gemma_config(self):
        return self.get('gemma')
