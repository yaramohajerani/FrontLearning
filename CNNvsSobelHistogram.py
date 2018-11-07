#!/anaconda2/bin/python2.7
u"""
CNNvsSobelHistogram.py
by Michael Wood (Last Updated by Yara Mohajerani 10/2018)

find path of least resistance through an image

Update History
    11/2018 - Yara: Don't separate train or test inputs based on glacier. Input subdir
                and get glacier name from spreadsheet
                Fix distance bug in frontComparisonErrors
                Add option for number of line segments for RMS
    10/2018 - Yara: Change input folder to be consistent with
                other scripts
    09/2018 - Yara: Clean up and add user input
    09/2018 - Michael: written
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from PIL import Image
import getopt


#############################################################################################
# All of the functions are run here
#-- main function to get user input and make training data
def main():
    #-- Read the system arguments listed after the program
    long_options = ['subdir=','method=','step=','indir=','interval=']
    optlist,arglist = getopt.getopt(sys.argv[1:],'=D:M:S:I:V:',long_options)

    subdir= 'all_data2_test'
    method = ''
    step = 50
    n_interval = 1000
    indir = ''
    for opt, arg in optlist:
        if opt in ('-D','--subdir'):
            subdir = arg
        elif opt in ('-M','--method'):
            method = arg
        elif opt in ('-S','--step'):
            step = np.int(arg)
        elif opt in ('-V','--interval'):
            n_interval = np.int(arg)
        elif opt in ('-I','--indir'):
            indir = os.path.expanduser(arg)


    #-- directory setup
    #- current directory
    current_dir = os.path.dirname(os.path.realpath(__file__))
    headDirectory = os.path.join(current_dir,'..','FrontLearning_data')

    glaciersFolder=os.path.join(headDirectory,'Glaciers')
    results_dir = os.path.join(headDirectory,'Results', subdir)

    #-- if user input not given, set label folder
    #-- else if input directory is given, then set the method based on that
    if indir == '':
        indir = os.path.join(results_dir,method,method)
    else:
        method = os.path.basename(indir)
        if method=='':
            sys.exit("Please do not put '/' at the end of indir.")

    print('input directory ONLY for NN output:%s'%indir)
    print('METHOD:%s'%method)

    outputFolder= os.path.join(results_dir,'Histograms/'+method+'_'+str(step)+'_%isegs'%n_interval)
    #-- make output folders
    if (not os.path.isdir(outputFolder)):
        os.mkdir(outputFolder)

    
    cnnPixelFolder = os.path.join(results_dir,method,method+' Pixel CSVs '+str(step))
    sobelPixelFolder = os.path.join(results_dir,'Sobel/Sobel Pixel CSVs '+str(step))

    cnnFrontFolder = os.path.join(results_dir,method,method+' Geo CSVs '+str(step))
    sobelFrontFolder = os.path.join(results_dir,'Sobel/Sobel Geo CSVs '+str(step))


    def seriesToNPoints(series,N):
        #find the total length of the series
        totalDistance=0
        for s in range(len(series[:,0])-1):
            totalDistance+=((series[s,0]-series[s+1,0])**2+(series[s,1]-series[s+1,1])**2)**0.5
        intervalDistance=totalDistance/(N-1)

        #make the list of points
        newSeries=series[0,:]
        currentS = 0
        currentPoint1=series[currentS,:]
        currentPoint2=series[currentS+1,:]
        for p in range(N-2):
            distanceAccrued = 0
            while distanceAccrued<intervalDistance:
                currentLineDistance=((currentPoint1[0]-currentPoint2[0])**2+(currentPoint1[1]-currentPoint2[1])**2)**0.5
                if currentLineDistance<intervalDistance-distanceAccrued:
                    distanceAccrued+=currentLineDistance
                    currentS+=1
                    currentPoint1 = series[currentS, :]
                    currentPoint2 = series[currentS + 1, :]
                else:
                    distance=intervalDistance-distanceAccrued
                    newX=currentPoint1[0]+(distance/currentLineDistance)*(currentPoint2[0]-currentPoint1[0])
                    newY = currentPoint1[1] + (distance / currentLineDistance) * (currentPoint2[1] - currentPoint1[1])
                    distanceAccrued=intervalDistance+1
                    newSeries=np.vstack([newSeries,np.array([newX,newY])])
                    currentPoint1=np.array([newX,newY])
        newSeries = np.vstack([newSeries, series[-1,:]])
        return(newSeries)

    def frontComparisonErrors(front1,front2):
        errors=[]
        for ff in range(len(front1)):
            dist=((front1[ff,0]-front2[ff,0])**2+(front1[ff,1]-front2[ff,1])**2)**0.5
            if not np.isnan(dist):
                errors.append(dist)
        return(errors)

    def rmsError(error):
        sum=0
        denom=0
        for e in error:
            if not np.isnan(e):
                sum+=e**2
                denom+=1
        RMS=(sum/denom)**0.5
        return(RMS)

    def generateLabelList(labelFolder):
        labelList=[]
        for fil in os.listdir(labelFolder):
            if fil[-6:] == 'B8.png' or fil[-6:] == 'B2.png':
                labelList.append(fil[:-4])
        return(labelList)

    # get glacier names
    def getGlacierList(labelList):
        f=open(os.path.join(glaciersFolder,'Scene_Glacier_Dictionary.csv'),'r')
        lines=f.read()
        f.close()
        lines=lines.split('\n')
        glacierList = []
        for sceneID in labelList:
            for line in lines:
                line=line.split(',')
                if line[0]==sceneID:   
                    glacierList.append(line[1])
        return(glacierList)

    #code to get the list of fronts and their images
    def getFrontList(glacierList,labelList):
        frontsList = []
        for ind,label in enumerate(labelList):
            glacier = glacierList[ind]
            f=open(os.path.join(glaciersFolder, glacier, '%s Image Data.csv'%glacier),'r')
            lines=f.read()
            f.close()
            lines=lines.split('\n')
            for line in lines:
                line=line.split(',')
                if line[1][:-4] == label:
                    frontsList.append(line[0])
        return(frontsList)

    labelList=generateLabelList(indir)
    glacierList=getGlacierList(labelList)
    frontList=getFrontList(glacierList,labelList)

    allCNNerrors=[]
    allSobelerrors=[]

    N=1
    N=len(labelList)
    for ll in range(N):
        glacier = glacierList[ll]
        label=labelList[ll]
        trueFrontFile=frontList[ll]

        ############################################################################
        # This section to get the front images
        trueImageFolder=headDirectory+'/Glaciers/'+glacier+'/Small Images'
        trueImage = Image.open(os.path.join(trueImageFolder,label+'_Subset.png')).transpose(Image.FLIP_LEFT_RIGHT).convert("L")

        cnnImageFolder = indir
        cnnImage = Image.open(os.path.join(cnnImageFolder,label + '_nothreshold.png')).transpose(Image.FLIP_LEFT_RIGHT).convert("L")

        sobelImageFolder = os.path.join(results_dir,'Sobel/Sobel')
        sobelImage = Image.open(os.path.join(sobelImageFolder,label + '.png')).transpose(Image.FLIP_LEFT_RIGHT).convert("L")

        ############################################################################
        # This section to get the front pixels

        # get the CNN front
        cnnPixelsFile = glacier + ' ' + label + ' Pixels.csv'
        cnnPixels = np.genfromtxt(cnnPixelFolder + '/' + cnnPixelsFile, delimiter=',')
        cnnPixels = seriesToNPoints(cnnPixels, n_interval)

        # get the Sobel front
        sobelPixelsFile = glacier + ' ' + label + ' Pixels.csv'
        sobelPixels = np.genfromtxt(sobelPixelFolder + '/' + sobelPixelsFile, delimiter=',')
        sobelPixels = seriesToNPoints(sobelPixels, n_interval)

        ############################################################################
        # This section to get the front data

        #get the true front
        trueFrontFolder = os.path.join(glaciersFolder,glacier,'Front Locations/3413')
        trueFront=np.genfromtxt(trueFrontFolder+'/'+trueFrontFile,delimiter=',')
        trueFront=seriesToNPoints(trueFront,n_interval)

        #get the CNN front
        cnnFrontFile=glacier+' '+label+' Profile.csv'
        cnnFront=np.genfromtxt(cnnFrontFolder+'/'+cnnFrontFile,delimiter=',')
        cnnFront=seriesToNPoints(cnnFront,n_interval)

        cnnErrors=frontComparisonErrors(trueFront,cnnFront)
        for error in cnnErrors:
            allCNNerrors.append(error)

        #get the Sobel front
        sobelFrontFile=glacier+' '+label+' Profile.csv'
        sobelFront=np.genfromtxt(sobelFrontFolder+'/'+sobelFrontFile,delimiter=',')
        sobelFront=seriesToNPoints(sobelFront,n_interval)

        frontXmin=np.min([np.min(trueFront[:,0]),np.min(cnnFront[:,0]),np.min(sobelFront[:,0])])
        frontXmax = np.max([np.max(trueFront[:, 0]), np.max(cnnFront[:, 0]), np.max(sobelFront[:, 0])])
        frontYmin = np.min([np.min(trueFront[:, 1]), np.min(cnnFront[:, 1]), np.min(sobelFront[:, 1])])
        frontYmax = np.max([np.max(trueFront[:, 1]), np.max(cnnFront[:, 1]), np.max(sobelFront[:, 1])])

        sobelErrors=frontComparisonErrors(trueFront,sobelFront)
        for error in sobelErrors:
            allSobelerrors.append(error)

        fig=plt.figure(figsize=(10,8))

        plt.subplot(2,3,1)
        plt.imshow(trueImage, cmap='gray')
        plt.gca().set_xlim([0, 200])
        plt.gca().set_ylim([300,0])
        plt.gca().axes.get_xaxis().set_ticks([])
        plt.gca().axes.get_yaxis().set_ticks([])
        plt.title('Original Image',fontsize=12)

        plt.subplot(2, 3, 2)
        plt.imshow(cnnImage, cmap='gray')
        plt.plot(cnnPixels[:, 0], cnnPixels[:, 1], 'y-',linewidth=3)
        plt.gca().set_xlim([0, 200])
        plt.gca().set_ylim([300, 0])
        plt.gca().axes.get_xaxis().set_ticks([])
        plt.gca().axes.get_yaxis().set_ticks([])
        plt.title('NN Solution',fontsize=12)

        plt.subplot(2, 3, 3)
        plt.imshow(sobelImage, cmap='gray')
        plt.plot(sobelPixels[:, 0], sobelPixels[:, 1], 'y-',linewidth=3)
        plt.gca().set_xlim([0, 200])
        plt.gca().set_ylim([300, 0])
        plt.gca().axes.get_xaxis().set_ticks([])
        plt.gca().axes.get_yaxis().set_ticks([])
        plt.title('Sobel Solution',fontsize=12)

        plt.subplot(2,3,4)
        plt.title('Geocoded Solutions',fontsize=12)
        plt.ylabel('Northing (km)',fontsize=12)
        plt.xlabel('Easting (km)',fontsize=12)
        plt.plot(trueFront[:,0]/1000,trueFront[:,1]/1000,'k-',label='True')
        plt.plot(cnnFront[:,0]/1000,cnnFront[:,1]/1000,'b-',label='NN')
        plt.plot(sobelFront[:,0]/1000,sobelFront[:,1]/1000,'g-',label='Sobel')
        plt.gca().set_xlim([frontXmin/1000,frontXmax/1000])
        plt.gca().set_ylim([frontYmin/1000, frontYmax/1000])
        plt.gca().set_xticks([frontXmin/1000,frontXmax/1000])
        plt.gca().set_yticks([frontYmin / 1000, frontYmax / 1000])
        plt.legend(loc=0)

        plt.subplot(2,3,5)
        plt.title('NN Errors Histogram',fontsize=12)
        bins=range(0,5000,100)
        y1, x1, _ =plt.hist(cnnErrors,alpha=0.5,color='blue',bins=bins,label='NN')
        plt.xlabel('RMS Error = '+'{0:.2f}'.format(rmsError(cnnErrors))+' m',fontsize=12)

        plt.subplot(2, 3, 6)
        plt.title('Sobel Error Histogram',fontsize=12)
        bins = range(0, 5000, 100)
        y2, x2, _ =plt.hist(sobelErrors, alpha=0.5, color='green', bins=bins, label='Sobel')
        plt.xlabel('RMS Error = ' + '{0:.2f}'.format(rmsError(sobelErrors)) + ' m',fontsize=12)

        plt.subplot(2,3,5)
        plt.gca().set_ylim([0,np.max([y1, y2])])
        plt.gca().set_xlim([0, np.max([x1, x2])])

        plt.subplot(2, 3, 6)
        plt.gca().set_ylim([0, np.max([y1, y2])])
        plt.gca().set_xlim([0, np.max([x1, x2])])

        plt.savefig(outputFolder + '/' + label + '.png',bbox_inches='tight')
        plt.close(fig)

    fig=plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.title(r"$\bf{e)}$" + " NN Error Histogram",fontsize=12)
    bins=range(0,5000,100)
    y1, x1, _ =plt.hist(allCNNerrors,alpha=0.5,color='blue',bins=bins,label='NN')
    plt.xlabel('RMS Error = '+'{0:.2f}'.format(rmsError(allCNNerrors))+' m',fontsize=12)
    plt.ylabel('Count (100 m bins)',fontsize=12)

    plt.subplot(1,2,2)
    plt.title(r"$\bf{f)}$" + " Sobel Error Histogram",fontsize=12)
    bins = range(0, 5000, 100)
    y2, x2, _ =plt.hist(allSobelerrors, alpha=0.5, color='green', bins=bins, label='Sobel')
    plt.xlabel('RMS Error = ' + '{0:.2f}'.format(rmsError(allSobelerrors)) + ' m',fontsize=12)

    plt.subplot(1,2,1)
    plt.gca().set_ylim([0,np.max([y1, y2])])
    plt.gca().set_xlim([0, np.max([x1, x2])])

    plt.subplot(1,2,2)
    plt.gca().set_ylim([0, np.max([y1, y2])])
    plt.gca().set_xlim([0, np.max([x1, x2])])

    plt.savefig(results_dir + '/Figure_4_'+'_'.join(method.split())+'_'+str(step)+'_%isegs'%n_interval+'.pdf',bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()