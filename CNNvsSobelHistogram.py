#code to make a histogram of the errors between the true, cnn, and sobel fronts

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
    long_options = ['glaciers=','method=','step=']
    optlist,arglist = getopt.getopt(sys.argv[1:],'=G:M:S:',long_options)

    glacier= 'Helheim'
    method = 'CNN'
    step = 50
    for opt, arg in optlist:
        if opt in ('-G','--glaciers'):
            glacier = arg
        elif opt in ('-M','--method'):
            method = arg
        elif opt in ('-S','--step'):
            step = np.int(arg)


    #-- directory setup
    #- current directory
    current_dir = os.path.dirname(os.path.realpath(__file__))
    headDirectory = os.path.join(current_dir,'..','FrontLearning_data')

    glaciersFolder=headDirectory+'/Glaciers'

    labelFolder=headDirectory+'/Results/'+glacier+' Results/'+method+'/'+method

    postProcessedOutputFolder=headDirectory+'/Results/'+glacier+' Results/'+method+'/'+method+' Post-Processed '+str(step)
    csvOutputFolder=headDirectory+'/Results/'+glacier+' Results/'+method+'/'+method+' Geo CSVs '+str(step)
    pixelOutputFolder=headDirectory+'/Results/'+glacier+' Results/'+method+'/'+method+' Pixel CSVs '+str(step)
    shapefileOutputFolder=headDirectory+'/Results/'+glacier+' Results/'+method+'/'+method+' Shapefile '+str(step)

    outputFolder=headDirectory+'/Results/'+glacier+' Results/Histograms/'+method+'_'+str(step)
    #-- make output folders
    if (not os.path.isdir(outputFolder)):
        os.mkdir(outputFolder)

    labelFolder=headDirectory+'/Results/'+glacier+' Results/'+method+'/'+method

    cnnPixelFolder = headDirectory+'/Results/' + glacier + ' Results/'+method+'/'+method+' Pixel CSVs '+str(step)
    sobelPixelFolder = headDirectory+'/Results/' + glacier + ' Results/Sobel/Sobel Pixel CSVs '+str(step)

    trueFrontFolder = headDirectory+'/Glaciers/' + glacier + '/Front Locations/3413'
    cnnFrontFolder = headDirectory+'/Results/' + glacier + ' Results/'+method+'/'+method+' Geo CSVs '+str(step)
    sobelFrontFolder = headDirectory+'/Results/' + glacier + ' Results/Sobel/Sobel Geo CSVs '+str(step)


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
            dist=((front1[ff,0]-front2[ff,0])**2+(front1[ff,1]-front2[ff,1]))**0.5
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

    #code to get the list of fronts and their images
    def getFrontList(glacier,labelList):
        metadataFile=headDirectory+'/Glaciers/'+glacier+'/'+glacier+' Image Data.csv'
        f=open(metadataFile)
        lines=f.read()
        f.close()
        lines=lines.split('\n')
        frontsList = []
        for label in labelList:
            for line in lines:
                line=line.split(',')
                if line[1][:-4] == label:
                    frontsList.append(line[0])
        return(frontsList)

    labelList=generateLabelList(labelFolder)
    frontList=getFrontList(glacier,labelList)

    allCNNerrors=[]
    allSobelerrors=[]

    N=1
    N=len(labelList)
    for ll in range(N):
        label=labelList[ll]
        trueFrontFile=frontList[ll]

        ############################################################################
        # This section to get the front images
        trueImageFolder=headDirectory+'/Glaciers/'+glacier+'/Small Images'
        trueImage = Image.open(trueImageFolder+'/'+label+'_Subset.png').transpose(Image.FLIP_LEFT_RIGHT).convert("L")

        cnnImageFolder = headDirectory+'/Results/'+glacier+' Results/'+method+'/'+method
        cnnImage = Image.open(cnnImageFolder + '/' + label + '_nothreshold.png').transpose(Image.FLIP_LEFT_RIGHT).convert("L")

        sobelImageFolder = headDirectory+'/Results/'+glacier+' Results/Sobel/Sobel'
        sobelImage = Image.open(sobelImageFolder + '/' + label + '.png').transpose(Image.FLIP_LEFT_RIGHT).convert("L")

        ############################################################################
        # This section to get the front pixels

        # get the CNN front
        cnnPixelsFile = glacier + ' ' + label + ' Pixels.csv'
        cnnPixels = np.genfromtxt(cnnPixelFolder + '/' + cnnPixelsFile, delimiter=',')
        cnnPixels = seriesToNPoints(cnnPixels, 100)

        # get the Sobel front
        sobelPixelsFile = glacier + ' ' + label + ' Pixels.csv'
        sobelPixels = np.genfromtxt(sobelPixelFolder + '/' + sobelPixelsFile, delimiter=',')
        sobelPixels = seriesToNPoints(sobelPixels, 100)

        ############################################################################
        # This section to get the front data

        #get the true front
        trueFront=np.genfromtxt(trueFrontFolder+'/'+trueFrontFile,delimiter=',')
        trueFront=seriesToNPoints(trueFront,100)

        #get the CNN front
        cnnFrontFile=glacier+' '+label+' Profile.csv'
        cnnFront=np.genfromtxt(cnnFrontFolder+'/'+cnnFrontFile,delimiter=',')
        cnnFront=seriesToNPoints(cnnFront,100)

        cnnErrors=frontComparisonErrors(trueFront,cnnFront)
        for error in cnnErrors:
            allCNNerrors.append(error)

        #get the Sobel front
        sobelFrontFile=glacier+' '+label+' Profile.csv'
        sobelFront=np.genfromtxt(sobelFrontFolder+'/'+sobelFrontFile,delimiter=',')
        sobelFront=seriesToNPoints(sobelFront,100)

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

    outputFolder = headDirectory+'/Results/Helheim Results/'
    plt.savefig(outputFolder + '/Figure_4_'+'_'.join(method.split())+'_'+str(step)+'.pdf',bbox_inches='tight')

if __name__ == '__main__':
    main()