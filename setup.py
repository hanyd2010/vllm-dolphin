# https://docs.vllm.ai/en/stable/contributing/model/registration.html
# https://docs.vllm.ai/en/stable/design/plugin_system.html

from setuptools import setup

setup(name='vllm_dolphin',
      version='0.1',
      packages=['vllm_dolphin'],
      install_requires=[
          "vllm>=0.9.0",
          "vllm-mbart>=0.1"
      ],
      entry_points={
          'vllm.general_plugins':
              ["register_dolphin = vllm_dolphin:register"]
      })
