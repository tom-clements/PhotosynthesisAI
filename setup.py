from setuptools import setup

setup(
   name='PhotosynthesisAI',
   version='1.0',
   description='AI for the board game Photosynthesis',
   author='Tom Clements',
   author_email='thomasalbertclements@gmail.com',
   packages=['PhotosynthesisAI'],  #same as name
   install_requires=['numpy', 'hexy', 'matplotlib'], #external packages as dependencies
)