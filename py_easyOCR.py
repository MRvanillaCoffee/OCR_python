import easyocr as eo
import torch
torch.cuda.is_available()
reader = eo.Reader(['en','th'])
result = reader.readtext('data/th_01.jpg')
print(result)