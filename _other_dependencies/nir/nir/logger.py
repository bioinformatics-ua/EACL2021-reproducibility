import logging

# Setup logging

#log = logging.getLogger('')#logging.basicConfig(filename='app.log', format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fhan = logging.FileHandler("notebooks.log", mode="a")
fhan.setLevel(logging.DEBUG)
logger.addHandler(fhan)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fhan.setFormatter(formatter)

log = logger
