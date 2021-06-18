from src.mcu.detectfaces import detectfaces
import argparse
import time
import os

def main(args):
	#filepath = args.impath
	#assert os.path.isfile(filepath), "Image path not valid (file does not exist)"
	dirpath = args.dirpath
	assert os.path.isdir(dirpath), "Path not valid (dir does not exist)"

	if dirpath is not "no":
		for filename in os.listdir(dirpath):
			filepath = os.path.join(dirpath, filename)
			if os.path.isfile(filepath):
				print(filepath)
				detectfaces(filepath)
				time.sleep(5)
	else:
		detectfaces(filepath)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-f', dest='impath', type=str, help="Input image path")
	parser.add_argument('-d', dest='dirpath', type=str, default="no", help="Input image path")
	
	args = parser.parse_args()
	main(args)