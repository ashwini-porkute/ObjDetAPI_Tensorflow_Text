import os,sys
import pandas as pd
import xml.etree.ElementTree as ET

def xml_to_csv(xml_file):
    xml_list = []
    print(">>> ", xml_file)
    img_names = []
    tree = ET.parse(xml_file)
    root = tree.getroot()
    c = 0
    for image in root.findall('image'):
      image_name = image[0].text
      img_name = image_name.split('/')[-1]
      img_names.append(img_name)
      c += 1
      #print(image_name)
      reso_x = int(image[3].get('x'))
      reso_y = int(image[3].get('y'))
      #print(reso_x,reso_y)
      for i in image[4]:
       #print(i.get('height'),i.get('width'),i.get('x'),i.get('y'))
        xmin = int(i.get('x'))
        ymin = int(i.get('y'))
        xmax = int(i.get('x')) + int(i.get('width'))
        ymax = int(i.get('y')) + int(i.get('height'))
        #print xmin,ymin,xmax,ymax
        value = (image_name,reso_x,reso_y,'text',xmin,ymin,xmax,ymax)
        xml_list.append(value)                     
      #print('\n')
    #print(xml_list)
    print('No of instances {}.xml = {}'.format(name,len(xml_list)))
    print('No. of images in {} xml = {}'.format(name,c))
   # sys.exit()
        
        
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df,img_names
        
print(' *** script started...!!!\n')
list1 = []

img_folder_path = 'img'
xmls_folder_path = 'xmls/'

img_list = os.listdir(img_folder_path)                

for xml in os.listdir(xmls_folder_path):
  #print(xml)
  xml_file = xmls_folder_path + xml

  name = xml.split('.')[0]

  xml_df,img_names = xml_to_csv(xml_file)
  for i in img_names:
    list1.append(i)
  xml_df.to_csv(str(name) + '_labels.csv', index=None)
#print len(list1)
for i in img_list:
  if i not in list1:
    print ('\nNOTE : extra image in img folder : ',i)
print('\n *** script executed...!!!')


