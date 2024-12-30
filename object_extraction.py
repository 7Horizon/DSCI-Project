# %% [code]
#!/usr/bin/env python3
try:
    import os
    import sys
    import math
    import numpy as np
    from matplotlib.colors import rgb_to_hsv

    import logging
    from skimage.transform import resize
    from skimage.io import imsave
    from skimage.morphology import erosion,closing,area_closing,closing,opening,dilation
    from skimage.morphology import square
    from skimage.measure import label, regionprops, regionprops_table
    import pdb
    import matplotlib.pyplot as plt
    import argparse
    from glob import glob
    import configparser
    from progressbar import progressbar
    from multiprocessing import Pool
    from functools import partial

except  ModuleNotFoundError as err:
    print("""Einige Module konnten nicht importiert werden. Um dieses Skript zu starten, sollte zunächst der Befehl "source bin/activate" (oder poetry shell o.ä.) in der Kommandozeile eingegeben werden.""")
    raise
    #sys.exit(1)

def intOrNone(n):
    if n is None:
        return None
    else: return int(n)

def load_file(imstr):
    im = plt.imread(imstr)
    if len(im.shape)==2:
        im = np.stack((im,)*3, axis=-1)
    elif im.shape[2]==1:
        im = np.stack((im,)*3, axis=-1).reshape(*im.shape,3)
    elif im.shape[2]==4:
        im = im[:,:,[0,1,2]]

    return im

def fill_borders(im,value_to_fill = False,fraction_of_rows_to_remove=0.0, fraction_of_cols_to_remove=0.1):
    """
    fraction_of_rows_to_remove: Der Bruchteil jeder Anzahl Bildzeilen, die sowohl oben wie auch unten am Bildrand abgeschnitten werden soll.
    fraction_of_cols_to_remove: Der Bruchteil jeder Anzahl Bildspalten, die sowohl links wie auch rechts am Bildrand abgeschnitten werden soll.

    """
    #    mask[:100,:]=False
#    mask[-100:,:]=False
    nRows,nCols = im.shape[0],im.shape[1]
    im_cropped = im.copy()
    nrow = int(fraction_of_rows_to_remove*nRows)
    ncol = int(fraction_of_cols_to_remove*nCols)
    logging.debug(f'Lösche {nrow} Zeilen und {ncol} Spalten.')

    if len(im_cropped.shape)==4 and im_cropped.shape[0]==1:
         im_cropped = im_cropped.reshape(*im_cropped.shape[1:])

    if len(im_cropped.shape)==2:
        im_cropped[:nrow,:]=value_to_fill
        if nrow>0:
            im_cropped[-nrow:,:]=value_to_fill

        im_cropped[:,:ncol]=value_to_fill
        if ncol>0:
            im_cropped[:,-ncol:]=value_to_fill
    elif len(im_cropped.shape)==3:
        im_cropped[:nrow,:,:]=value_to_fill
        if nrow>0:
            im_cropped[-nrow:,:,:]=value_to_fill

        im_cropped[:,:ncol,:]=value_to_fill
        if ncol>0:
            im_cropped[:,-ncol:,:]=value_to_fill

    else:
        raise AssertionError('Dieser Fall wurde noch nicht beachtet. Nur Graustufen- und Farbbilder werden unterstützt.')

    return im_cropped

def fill_borders_separate(im, value_to_fill=0,
                 top_fraction=0.0, bottom_fraction=0.0,
                 left_fraction=0.1, right_fraction=0.1):
    """
    top_fraction: Der Bruchteil der Bildzeilen, die oben abgeschnitten und ersetzt werden sollen.
    bottom_fraction: Der Bruchteil der Bildzeilen, die unten abgeschnitten und ersetzt werden sollen.
    left_fraction: Der Bruchteil der Bildspalten, die links abgeschnitten und ersetzt werden sollen.
    right_fraction: Der Bruchteil der Bildspalten, die rechts abgeschnitten und ersetzt werden sollen.
    """
    nRows, nCols = im.shape[0], im.shape[1]
    im_cropped = im.copy()

    # Berechne die Anzahl der Zeilen und Spalten, die ersetzt werden sollen
    n_top = int(top_fraction * nRows)
    n_bottom = int(bottom_fraction * nRows)
    n_left = int(left_fraction * nCols)
    n_right = int(right_fraction * nCols)

    logging.debug(f'Lösche {n_top} Zeilen oben, {n_bottom} Zeilen unten, {n_left} Spalten links und {n_right} Spalten rechts.')

    if len(im_cropped.shape) == 4 and im_cropped.shape[0] == 1:
        im_cropped = im_cropped.reshape(*im_cropped.shape[1:])

    if len(im_cropped.shape) == 2:
        # Für Graustufenbilder
        im_cropped[:n_top, :] = value_to_fill
        if n_bottom > 0:
            im_cropped[-n_bottom:, :] = value_to_fill

        im_cropped[:, :n_left] = value_to_fill
        if n_right > 0:
            im_cropped[:, -n_right:] = value_to_fill

    elif len(im_cropped.shape) == 3:
        # Für Farbbilder
        im_cropped[:n_top, :, :] = value_to_fill
        if n_bottom > 0:
            im_cropped[-n_bottom:, :, :] = value_to_fill

        im_cropped[:, :n_left, :] = value_to_fill
        if n_right > 0:
            im_cropped[:, -n_right:, :] = value_to_fill

    else:
        raise AssertionError('Dieser Fall wurde noch nicht berücksichtigt. Nur Graustufen- und Farbbilder werden unterstützt.')

    return im_cropped

def ask_path_creation(path,yes_to_all=False):
    """
    Fragt den Nutzer, ob das Verzeichnis <path> erstellt werden darf (falls es nicht schon existiert).
    """
    if not os.path.exists(path):
            logging.info(f"Der Pfad {path} konnte nicht gefunden werden. ")
            if not yes_to_all:
                s = input("Darf er erstellt werden (j/N)?")
                if s.lower() in ['j','y','ja','yes']:
                    mkdir=True
                else:
                    mkdir=False
            else:
                mkdir=True
            if mkdir:
                os.makedirs(path,exist_ok=True)
                logging.info(f'Pfad {path} wurde erstellt')
                path_now_exists = True
            else:
                print('Abbruch.')
                path_now_exists = False 
    else:
        logging.info(f"Der Pfad {path} existiert bereits. ")
        if not yes_to_all:
            s = input("Weiterfahren (j/N)?")
            if not s.lower() in ['j','y','ja','yes']:
                print('Abbruch.')
                sys.exit(0)

        path_now_exists = True
    return path_now_exists 

    
def generate_mask_with_hsv_threshold(im, sign=None,hue_threshold=None, saturation_threshold=None, value_threshold=None):
    """
    Erstelle eine Maske des Bildes (aktuell mit einem Schwellwert auf dem Value-Kanal des Bildes).
    """
    assert hue_threshold or saturation_threshold or value_threshold, 'Mindestens einer der Schwellwerte muss gesetzt werden'
    if sign is not None: 
        print('Warning- sign is deprecated')
        value_sign=sign
        value_threshold = np.abs(value_threshold)*value_sign 
    
    im_hsv = rgb_to_hsv(im)*np.array([255,255,1])[np.newaxis,np.newaxis,:] 
    
    if hue_threshold:
        hue_sign = np.sign(hue_threshold)
        mask = hue_sign*im_hsv[:,:,0]>hue_threshold
    if saturation_threshold:
        saturation_sign= np.sign(saturation_threshold)
        mask = saturation_sign*im_hsv[:,:,1]>saturation_threshold
    if value_threshold:
        value_sign= np.sign(value_threshold)
        mask = value_sign*im_hsv[:,:,2]>value_threshold

    return mask

def morphology_transform(mask,shape = square,erosion_size=5,dilation_size=50):
    """
    Important morphological preprocessing TUNE HERE IF NECESSARY
    """


    mask = erosion(mask.copy(),shape(1))
    mask = dilation(mask,shape(1))

    mask = dilation(mask,shape(8))
    mask = erosion(mask,shape(8))

    return mask

def create_mask(im,top_fraction=0,bottom_fraction=0,left_fraction=0,right_fraction=0,hue_threshold=None,saturation_threshold=None,value_threshold=100,erosion_size=0,dilation_size=0):
    """
    Erstelle eine (binäre) Maske aus dem Bild im durch Randbeschneidung, HSV-Schwellwertbildung und morphologischen Transformationen. 
    Die Maske soll die relevanten Ausschnitte von im angeben.

    """
    #Schneide die Ränder des Bildes ab, wo sicher keine Objekte drin sind
    im_filled= fill_borders_separate(im,top_fraction=top_fraction,bottom_fraction=bottom_fraction,left_fraction=left_fraction,right_fraction=right_fraction)

    #Generiere eine Maske (binäres Bild), welche Vordergrundpixel mit True bezeichnet, den Hintergrund mit False
    mask = generate_mask_with_hsv_threshold(im_filled,hue_threshold=hue_threshold,saturation_threshold=saturation_threshold,value_threshold=value_threshold)

    #Passe die Maske mit morphologischen Transformationen an, insbesondere, um vereinzelte Pixel zu eliminieren (Erosion) und ev. um die Maske leicht zu vergrössern (Dilatation)
    if erosion_size > 0 and dilation_size > 0:
        morphed_mask = morphology_transform(mask,erosion_size=erosion_size,dilation_size=dilation_size)
    else:
        morphed_mask = mask
    return morphed_mask,im_filled

# def create_mask(im,fraction_of_rows_to_remove=0.0,fraction_of_cols_to_remove=0.1,hue_threshold=None,saturation_threshold=None,value_threshold=100,erosion_size=5,dilation_size=50):
#     """
#     Erstelle eine (binäre) Maske aus dem Bild im durch Randbeschneidung, HSV-Schwellwertbildung und morphologischen Transformationen.
#     Die Maske soll die  relevanten Ausschnitte von im angeben.
#
#     """
#     #Schneide die Ränder des Bildes ab, wo sicher keine Objekte drin sind
#     im_filled= fill_borders(im,value_to_fill=0,fraction_of_rows_to_remove=fraction_of_rows_to_remove,fraction_of_cols_to_remove=fraction_of_cols_to_remove)
#
#     #Generiere eine Maske (binäres Bild), welche Vordergrundpixel mit True bezeichnet, den Hintergrund mit False
#     mask = generate_mask_with_hsv_threshold(im_filled,hue_threshold=hue_threshold,saturation_threshold=saturation_threshold,value_threshold=value_threshold)
#
#     #Passe die Maske mit morphologischen Transformationen an, insbesondere, um vereinzelte Pixel zu eliminieren (Erosion) und ev. um die Maske leicht zu vergrössern (Dilatation)
#     morphed_mask = morphology_transform(mask,erosion_size=erosion_size,dilation_size=dilation_size)
#     return morphed_mask,im_filled
    
def create_masked_image(im,mask):
    """
    Erstelle mit der Maske eine Version von im, in welcher alle Hintergrundpixel auf den Wert 0 gesetzt werden. Schneide die Bildteile ausserhalb der Maske ab.
    """
    masked_image = np.where(mask,1,0)[:,:,np.newaxis]*im
    return masked_image

def extract_regions(label_img,masked_image,fraction_of_rows_to_remove=0):
    regions = regionprops(label_img,intensity_image=masked_image)
    Nrows,Ncols=masked_image.shape[:2]
    region_images_list=[]
    for props in regions:
        bx,by,minr,maxr,minc,maxc,orientation = compute_bb(props)
        if minr >= fraction_of_rows_to_remove*Nrows and maxr <=Nrows-fraction_of_rows_to_remove*Nrows:
            mi = masked_image[minr:maxr,minc:maxc]
            region_images_list.append(mi)
        #angle = 180*orientation/np.pi

    return region_images_list,regions
    
def show_extracted_regions(regionlist,axlist=None,fig=None):
    n=len(regionlist)
    if axlist is not None:
        assert n==len(axlist),'axlist and regionlist must have the same length'
    else:
        if fig is None:
            fig=plt.figure()
        if n>1:
            axlist = fig.subplots(1,n)
        elif n==1:
            axlist = [fig.subplots(1,n)]
        else: return None


    for reg,ax in zip(regionlist,axlist):
        ax.imshow(reg)
        ax.set_xticks([]); ax.set_yticks([]);
    return fig
    
def object_segmentation(morphed_mask,masked_image,fraction_of_rows_to_remove=0):
    """
    Erzeuge mit der Maske Objektsegmente und schneide diese in masked_image aus.
    """
    label_img = label(morphed_mask)

    regionlist,regions = extract_regions(label_img,masked_image,fraction_of_rows_to_remove=fraction_of_rows_to_remove)

    return regionlist,regions

    
def image_preprocessing(im,**kwargs):
    """
    Diese Routine enthält die ganze Vorverarbeitung eines Rohbildes. Rückgabe:
    regionlist und regions: enthalten Informationen über die gefundenen Objekte
    mask: Maske, mit welcher die Objekte ausgeschnitten wurden
    masked_image: Originalbild mit unterdrücktem Hintergrund
    """
    #Erstelle eine Maske für das Bild.
    morphed_mask,im_filled = create_mask(im,**kwargs)
        
    #Erstelle mit der Maske eine Version von im, in welcher alle Hintergrundpixel auf den Wert 0 gesetzt werden
    masked_image = create_masked_image(im_filled,morphed_mask)

    # Das Bild enthält nun nur noch relevante Pixel (alle anderen wurden ausmaskiert). Segmentiere nun das Bild, um Objekte zu erhalten.
    fraction_of_rows_to_remove = kwargs['top_fraction'] + kwargs['bottom_fraction']
    regionlist,regions = object_segmentation(morphed_mask,masked_image,fraction_of_rows_to_remove=fraction_of_rows_to_remove)

    return regionlist,regions,morphed_mask,masked_image

def insert_region_number(basename,ireg):
    """
    Gemäss Konvention können wir Dateinamen der Form <irgendwas>_<Klassenname>.jpg haben. 
    Die extrahierten Regionen sollen dann so heissen: <irgendwas>_<region_index>_<Klassenname>.jpg
    >>> insert_region_number('image_123_Hund.jpg',5)
    'image_123_5_Hund.jpg'
    """
    basename,ext = os.path.splitext(basename)

    start = basename.rfind('_')
    end = basename.rfind('.')
    return f'{basename[:start]}_{ireg}_{basename[start+1:]}{ext}'

    
def save_regionlist_to_folder(fn,regionlist,regions,outputpath,write_summary_file=False,min_num_pixels=0):
    """
    min_num_pixels: Kleinste Region (in Anzahl Pixel), welche gespeichert wird
    """
    fnpart = os.path.basename(fn)
    for ireg,(regA,region) in enumerate(zip(regionlist,regions)):
        #if region.image.shape[0]*region.image.shape[1]<=min_num_pixels: #region.num_pixels<= min_num_pixels, regA.shape[0]*regA.shape[1]?  
        if region.area <= min_num_pixels: #region.num_pixels<= min_num_pixels, regA.shape[0]*regA.shape[1]?
            logging.info(f'skipping region: num_pix={region.num_pixels},shape:{regA.shape},area:{region.area}, as num_pix is < {min_num_pixels}')
            continue
        #erstelle einen Dateinamen auf der Basis des Elternnamens, so dass allfällige Klassennamen immer noch an der richtigen Stelle (zuletzt, hinter dem letzten "_") stehen.
        region_filename = insert_region_number(fnpart,ireg)

        assert os.path.exists(outputpath),f'Der Pfad {outputpath} sollte existieren.'
        outfile = os.path.join(outputpath,region_filename)
        imsave(outfile,regA.astype('uint8'))

    if (len(regionlist)>1) and write_summary_file:
        outfile = os.path.join(outputpath,fnpart+'_all.jpg')
        
        fig = plt.figure()
        show_extracted_regions(regionlist,axlist=None,fig=fig)
        logging.info(f'writing to {outfile}')

        fig.savefig(outfile)
    return True 


def debug_extraction(im):
    """
    Show 4 Plots outlining the extraction process. For debugging purposes.
    """
    mask,im_filled = create_mask(im)
    masked_image = create_masked_image(im_filled,mask)
    regionlist,regions = object_segmentation(mask,im_filled)

    if len(regions)==0:
        print('Keine Objekte gefunden')
    fig,axlist = plt.subplots(1,4+len(regionlist),figsize=(15,5))
    axlist[0].imshow(im); axlist[0].set_title('original')

    mark_regions_in_masked_image(regions,masked_image,axlist=axlist[1:3],image1=im)
    axlist[3].imshow(im);axlist[3].set_xticks([]);axlist[3].set_yticks([])
    show_extracted_regions(regionlist,axlist=axlist[4:],fig=fig)
    if len(axlist)>=5:
        axlist[4].set_title('Regions')
    return fig

def compute_bb(props):
    y0, x0 = props.centroid
    orientation = props.orientation
    x1 = x0 + math.cos(orientation) * 0.5 * props.axis_minor_length
    y1 = y0 - math.sin(orientation) * 0.5 * props.axis_minor_length
    x2 = x0 - math.sin(orientation) * 0.5 * props.axis_major_length
    y2 = y0 - math.cos(orientation) * 0.5 * props.axis_major_length

    minr, minc, maxr, maxc = props.bbox
    bx = (minc, maxc, maxc, minc, minc)
    by = (minr, minr, maxr, maxr, minr)
    return bx,by,minr,maxr,minc,maxc,orientation




def show_region(region,ax=None):
    bx,by,minr,maxr,minc,maxc,orientation = compute_bb(region)
    angle = 180*orientation/np.pi

    if ax is None:
        plt.plot(bx, by, '-b', linewidth=2.5)
        plt.title(f'Orientation: {(180*orientation/np.pi):1.0f}°',va='center')

    else:
        ax.plot(bx, by, '-b', linewidth=2.5)
        ax.set_title(f'Orientation: {(180*orientation/np.pi):1.0f}°',va='center')


def mark_regions_in_masked_image(regions,masked_image,image1=None,axlist=None):
    if image1 is None:
        axlist[0].imshow(masked_image)
        axlist[0].set_title('original')
    else:
        axlist[0].imshow(masked_image)
    imglist=[]
    if axlist is None:
        fig, axlist = plt.subplots(1,2)
    else:
        assert len(axlist)==2,'axlist must have length 2'
    

    axlist[1].imshow(masked_image)
    
    for region in regions:
        show_region(region,ax=axlist[1])
    for ax in axlist:
        ax.set_xticks([]);ax.set_yticks([])
    

def write_config_file(top_fraction,bottom_fraction,left_fraction,right_fraction,hue_threshold,saturation_threshold,value_threshold,erosion_size,dilation_size,minimum_number_of_pixels,preprocessing_resolution,output_path,imgfns,write_summary_file,fn='learning_city_lab.config'):

    output_path = os.path.abspath(output_path)

    path_set = set([os.path.dirname(fn) for fn in imgfns])

    config = configparser.ConfigParser()
    config.add_section('object extraction')
    config.set('object extraction', 'top_fraction', f'{top_fraction:1.3f}')
    config.set('object extraction', 'bottom_fraction', f'{bottom_fraction:1.3f}')
    config.set('object extraction', 'left_fraction', f'{left_fraction:1.3f}')
    config.set('object extraction', 'right_fraction', f'{right_fraction:1.3f}')
    if not hue_threshold is None:
        config.set('object extraction', 'hue_threshold', f'{hue_threshold:3.0f}')
    else:
        config.set('object extraction', 'hue_threshold', 'None')
    if not saturation_threshold is None:
        config.set('object extraction', 'saturation_threshold', f'{saturation_threshold:3.0f}')
    else:
        config.set('object extraction', 'saturation_threshold', 'None')
    if not value_threshold is None:
        config.set('object extraction', 'value_threshold', f'{value_threshold:3.0f}')
    else:
        config.set('object extraction', 'value_threshold', 'None')
    config.set('object extraction', 'erosion size', f'{erosion_size}')
    config.set('object extraction', 'dilation size', f'{dilation_size}')
    config.set('object extraction', 'minimum region area', f'{minimum_number_of_pixels}')
    config.set('object extraction', 'output_path', f'{os.path.abspath(output_path)}')
    config.set('object extraction', 'preprocessing_resolution', f'{preprocessing_resolution}')
    if len(path_set)==1:
        input_path = os.path.abspath(list(path_set)[0])
        config.set('object extraction', 'object_extraction_input_path', f'{input_path}')
    else:
        config.set('object extraction', 'object_extraction_input_path', 'diverse')

    # Schreibe die Konfigurationsdatei
    fullfn = os.path.join(output_path,fn)
    with open(fullfn, 'w') as configfile:
        config.write(configfile)

    logging.info(f"config-Datei wurde nach {fullfn} geschrieben.")
    return fullfn

def process_file(fn,top_fraction=0,bottom_fraction=0,left_fraction=0,right_fraction=0,hue_threshold=None,saturation_threshold=None,value_threshold=100,erosion_size=0,dilation_size=0,min_num_pixels=0,outputpath='/tmp/',write_summary_file=True,preprocessing_resolution=None):
    """
    Objektextraktion aus einer Datei. <fn> ist eine Bilddatei. Die Schlüsselwortargumente werden an image_processing weitergeleitet.
    """

    im = load_file(fn)
    if preprocessing_resolution is not None:
        if preprocessing_resolution[0]< im.shape[0] and preprocessing_resolution[1]< im.shape[1]:
            im = (255*resize(im,preprocessing_resolution)).astype('uint8') #resize ergibt ein Bild im mit Wertebereich (0,1)
        else:
            logging.warning(f'preprocessing resolution {preprocessing_resolution} ist nicht kleiner als die ursprüngliche Bildgrösse {im.shape}. Bildgrösse wird nicht verändert')

    regionlist,regions,morphed_mask,masked_image = image_preprocessing(im,top_fraction=top_fraction,bottom_fraction=bottom_fraction,left_fraction=left_fraction,right_fraction=right_fraction,hue_threshold=hue_threshold,saturation_threshold=saturation_threshold,value_threshold=value_threshold,erosion_size=erosion_size,dilation_size=dilation_size)
    #        fig = debug_extraction(im)
    #        fig.savefig('debug.png')
    fnpart = os.path.splitext(os.path.basename(fn))[0]
    continue_the_loop = save_regionlist_to_folder(fn,regionlist,regions,outputpath,write_summary_file=write_summary_file,min_num_pixels=min_num_pixels)
    return continue_the_loop

def parse_arguments():
    parser = argparse.ArgumentParser(description='''Dieses Kommandozeilenprogramm verlangt ein Dateinamen-Argument "imgfn" (kurz für image file name). Es führt eine Objektlokalisation in diesem Bild durch und extrahiert den Bildausschnitt in den Ausgabepfad (welche mit -o angegeben werden kann). Ein Beispiel zum Ausprobieren ist mitgeliefert:\n
    python object_extraction.py testimgs_in/K001.jpg -o testimgs_out/
oder auch

    python object_extraction.py -o testimgs_out -fr 0 -fc 0.13 -th 110 testimgs_in/*.jpg 
    python object_extraction.py -o /tmp/object_extraction_Nudeln -fr 0 -fc 0.13 -th 110 "/mnt/c/Users/beat.toedtli/OST/IPM - DSAI/Learning_City_Lab_Bilder/1_object_extraction/Nudeln_single_in/*.jpg"
    python object_extraction.py -w -o testimgs_out/ -fr 0 -fc 0.13 -th 110 testimgs_in/*.jpg
    python object_extraction.py -o t150_10000_np -fr 0 -fc 0.13 -th 150 -es 10 -ds 50 -mpx 10000 ../0_Daten/Daten_Giraffe/Trainingsdaten/IMG_20230831_105436.jpg

''',
formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('imgfns',nargs='+',help='Bilddateien (*.png, *.jpg u.ä.) mit einer Aufnahme von Objekten vor schwarzem Hintergrund')
    parser.add_argument('-w','--write-summary-file',action='store_true',help='Gebe eine _all.jpg Datei aus, welche pro Bild die extrahierten Objekte zusammenfasst, wenn es zwei oder mehr sind.')
    parser.add_argument('-d','--debug',action='store_true',help='Erstelle eine Bilddatei debug.png, welche die Objektextraktionsschritte aufzeigt.')
    parser.add_argument('-o','--output',help='Ausgabepfad für extrahierte Objekte',default='./tmp')
    parser.add_argument('-tf','--top_fraction',help='Dieser Zahl (zwischen 0 und 1) definiert die Höhe des Randbandes als Bruchteil der Bildhöhe von oben. Objekte, deren Bounding-Box teilweise dieses Randband berühren, werden nicht extrahiert. Damit können objekte, die am Rand des Blickfeldes der Kamera auftreten, verworfen werden. Diese Objekte sind oft unvollständig und daher für einen Bildklassifikationstrainingsdatensatz nicht geeignet',default=0.1)
    parser.add_argument('-bf','--bottom_fraction',help='Dieser Zahl (zwischen 0 und 1) definiert die Höhe des Randbandes als Bruchteil der Bildhöhe von oben. Objekte, deren Bounding-Box teilweise dieses Randband berühren, werden nicht extrahiert. Damit können objekte, die am Rand des Blickfeldes der Kamera auftreten, verworfen werden. Diese Objekte sind oft unvollständig und daher für einen Bildklassifikationstrainingsdatensatz nicht geeignet',default=0.1)
    parser.add_argument('-lf','--left_fraction',help='Diese Zahl (zwischen 0 und 1) definiert die Anzahl Pixelspalten am linken Bildrand, welche vom Kamerabild abgeschnitten werden. Dort zeigen sich oft helle Bereiche ausserhalb des Förderbandes. Diese können als Objekte (falsch-)interpretiert werden, was oft unerwünscht ist.',default=0.13)
    parser.add_argument('-rf','--right_fraction',help='Diese Zahl (zwischen 0 und 1) definiert die Anzahl Pixelspalten am rechten Bildrand, welche vom Kamerabild abgeschnitten werden. Dort zeigen sich oft helle Bereiche ausserhalb des Förderbandes. Diese können als Objekte (falsch-)interpretiert werden, was oft unerwünscht ist.',default=0.13)

    parser.add_argument('-es','--erosion_size',help='Diese Zahl gibt die Grösse der Erosionsmaske an, welche auf die Bildmaske angewendet wird. Anschliessend wird eine Dilatation durchgeführt- siehe --dilation_size',default=5)
    parser.add_argument('-ds','--dilation_size',help='Diese Zahl gibt die Grösse der Dilatationsmakse an, welche auf die Bildmaske angewendet wird. Davor wird eine Erosion durchgeführt- siehe --erosion_size.',default=50)
    parser.add_argument('-mpx','--minimum_number_of_pixels',help='Die kleinstmögliche Anzahl Pixel einer Region, für die noch ein Ausschnitt generiert wird.',default=10000)

    parser.add_argument('-vth','--value_threshold',help='Vorzeichenbehafteter Schwellwert für Hintergrundpixel im Value-Kanal des Bildes, ein Schwellwert im Bereich [-255,255]. Bei positivem Schwellwert werden Pixel mit Werten nicht grösser als der value_threshold als Hintergrundpixel angesehen. Bei einem negativen Schwellwert werden Pixel mit Werten nicht kleiner als der Absolutwert des value_threshold als Hintergrundpixel angesehen.',default=100)
    parser.add_argument('-hth','--hue_threshold',help='Schwellwert für Hintergrundpixel im Hue-Kanal des Bildes, ein Schwellwert im Bereich [0,255]. Werte nicht grösser als der hue_threshold werden als Hintergrundpixel angesehen. Negative Werte sind möglich, siehe --value_threshold.',default=None)
    parser.add_argument('-sth','--saturation_threshold',help='Schwellwert für Hintergrundpixel im Saturation-Kanal des Bildes, ein Schwellwert im Bereich [0,255]. Werte nicht grösser als der saturation_threshold werden als Hintergrundpixel angesehen. Negative Werte sind möglich, siehe --value_threshold.',default=None)
    parser.add_argument('-n','--num_cores',help='Anzahl Kerne, die zur Berechnung benützt werden. So lässt die Rechenzeit reduzieren.',default=1)
    parser.add_argument('-pr','--preprocessing_resolution',help='Reduziere die Auflösung der geladenen Bilder auf dieses Tupel (Anzahl Zeilen,Anzahl Spalten). Eingabe ist Komma-separiert, Klammern werden entfernt',default='1000,1000')
    parser.add_argument('-y','--yes_to_all',action='store_true',help='Keine Rückfragen',default=False)
    args = parser.parse_args()

    return args

def process_arguments(args):
    yta = args.yes_to_all
    outputpath= args.output
    outputpath_Ausschnitte = os.path.join(outputpath,'Ausschnitte')
    ask_path_creation(outputpath_Ausschnitte,yes_to_all=yta)

    top_fraction = float(args.top_fraction)
    bottom_fraction = float(args.bottom_fraction)
    left_fraction = float(args.left_fraction)
    right_fraction = float(args.right_fraction)



    preprocessing_resolution = [int(f) for f in args.preprocessing_resolution.split(',')]
    print('preprocessing_resolution ',preprocessing_resolution )
    erosion_size= intOrNone(args.erosion_size)
    dilation_size= intOrNone(args.dilation_size)
    hue_threshold = intOrNone(args.hue_threshold)
    value_threshold = intOrNone(args.value_threshold)
    saturation_threshold = intOrNone(args.saturation_threshold)
    write_summary_file = args.write_summary_file
    num_cores = int(args.num_cores)
    minimum_number_of_pixels = int(args.minimum_number_of_pixels)


    # Nehme das Bild <fn> (erstes Argument dieses Skripts)
    if len(args.imgfns)>1:
        imgfns = [item for arg in args.imgfns for item in glob(arg)]
    else:
        imgfns = [item for item in glob(args.imgfns[0])] 

    full_config_filename = write_config_file(top_fraction,bottom_fraction,left_fraction,right_fraction,hue_threshold,saturation_threshold,value_threshold,erosion_size,dilation_size,minimum_number_of_pixels,preprocessing_resolution,outputpath,imgfns,write_summary_file ,fn='object_extraction.config')

    kwargs = {'top_fraction':top_fraction,'bottom_fraction':bottom_fraction,'left_fraction':left_fraction,'right_fraction':right_fraction,'hue_threshold':hue_threshold,'saturation_threshold':saturation_threshold,'value_threshold':value_threshold,
            'erosion_size':erosion_size,'dilation_size':dilation_size,'min_num_pixels':minimum_number_of_pixels,'outputpath':outputpath_Ausschnitte,'preprocessing_resolution':preprocessing_resolution}
    return imgfns,kwargs

def main(imgfns,kwargs):
    if kwargs['num_cores']>1:
        with Pool(kwargs['num_cores']) as p:
            # partial erstellt eine Funktion mit einem einzigen Argument, dem Dateinamen
            # p.map verteilt die Berechnung auf num_cores Kerne
            del kwargs['num_cores']
            if 'write_summary_file' in kwargs: del kwargs['write_summary_file']
            single_argument_process_file = partial(process_file, **kwargs)
            retVals = p.map(single_argument_process_file, imgfns)
    else:
        del kwargs['num_cores']
        if 'write_summary_file' in kwargs: del kwargs['write_summary_file']
        for fn in progressbar(imgfns):
            continue_the_loop = process_file(fn,**kwargs)

            if not continue_the_loop: 
                break
    outputpath_Ausschnitte = kwargs['outputpath']
    logging.info(f'Die Bildausschnitte wurden hierher kopiert: {outputpath_Ausschnitte}')
   