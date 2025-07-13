import torch
import sys
import pprint

def main(pth_file):
  data = torch.load(pth_file, map_location='cpu')
  pprint.pprint(data)

if __name__ == "__main__":
  if len(sys.argv) != 2:
    print(f"Usage: python {sys.argv[0]} <file.pth>")
    sys.exit(1)
  main(sys.argv[1])