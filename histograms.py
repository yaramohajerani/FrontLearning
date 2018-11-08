#!/anaconda2/bin/python2.7
u"""
histograms.py
by Yara Mohajerani (Last Update 11/2018)
Forked from CNNvsSobelHistogram.py by Michael Wood 

find path of least resistance through an image and quantify errors

Update History
    11/2018 - Forked from CNNvsSobelHistogram.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from PIL import Image
import getopt
import copy


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

    outputFolder= os.path.join(results_dir,'Histograms',method+'_'+str(step)+'_%isegs'%n_interval)
    #-- make output folders
    if (not os.path.isdir(outputFolder)):
        os.mkdir(outputFolder)

    datasets = ['NN','Sobel','Manual']
    
    pixelFolder = {}
    frontFolder = {}

    pixelFolder['NN'] = os.path.join(results_dir,method,method+' Pixel CSVs '+str(step))
    pixelFolder['Sobel'] = os.path.join(results_dir,'Sobel/Sobel Pixel CSVs '+str(step))
    pixelFolder['Manual'] = os.path.join(results_dir,'output_handrawn/output_handrawn Pixel CSVs '+str(step))

    frontFolder['NN'] = os.path.join(results_dir,method,method+' Geo CSVs '+str(step))
    frontFolder['Sobel'] = os.path.join(results_dir,'Sobel/Sobel Geo CSVs '+str(step))
    frontFolder['Manual'] = os.path.join(results_dir,'output_handrawn/output_handrawn Geo CSVs '+str(step))

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

    allerrors = {}
    allerrors['NN']=[]
    allerrors['Sobel']=[]
    allerrors['Manual']=[]

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

        frontImageFolder = {}
        frontImageFolder['NN'] = indir
        frontImageFolder['Sobel'] = os.path.join(results_dir,'Sobel/Sobel')
        frontImageFolder['Manual'] = os.path.join(os.path.dirname(indir),'output_handrawn')

        frontImage = {}
        pixels = {}
        for d,tl in zip(datasets,['_nothreshold','','_nothreshold']):
            frontImage[d] = Image.open(os.path.join(frontImageFolder[d],label \
                    + '%s.png'%tl)).transpose(Image.FLIP_LEFT_RIGHT).convert("L")
            ############################################################################
            # This section to get the front pixels

            # get the front
            pixelsFile = glacier + ' ' + label + ' Pixels.csv'
            pixels[d] = np.genfromtxt(os.path.join(pixelFolder[d],pixelsFile), delimiter=',')
            pixels[d] = seriesToNPoints(pixels[d], n_interval)


        ############################################################################
        # This section to get the front data

        #get the true front
        trueFrontFolder = os.path.join(glaciersFolder,glacier,'Front Locations/3413')
        trueFront=np.genfromtxt(trueFrontFolder+'/'+trueFrontFile,delimiter=',')
        trueFront=seriesToNPoints(trueFront,n_interval)

        front = {}
        errors = {}
        for d in datasets:
            #get the front
            frontFile=glacier+' '+label+' Profile.csv'
            front[d]=np.genfromtxt(os.path.join(frontFolder[d],frontFile),delimiter=',')
            front[d]=seriesToNPoints(front[d],n_interval)

            errors[d]=frontComparisonErrors(trueFront,front[d])
            for error in errors[d]:
                allerrors[d].append(error)


        frontXmin = np.min(np.concatenate(([np.min(trueFront[:, 0])], [np.min(front[d][:,0]) for d in datasets])))
        frontXmax = np.max(np.concatenate(([np.max(trueFront[:, 0])], [np.max(front[d][:, 0]) for d in datasets])))
        frontYmin = np.min(np.concatenate(([np.min(trueFront[:, 1])], [np.min(front[d][:, 1]) for d in datasets])))
        frontYmax = np.max(np.concatenate(([np.max(trueFront[:, 1])], [np.max(front[d][:, 1]) for d in datasets])))


        fig=plt.figure(figsize=(10,8))

        n_panels = len(front)+1
        plt.subplot(2,n_panels,1)
        plt.imshow(trueImage, cmap='gray')
        plt.gca().set_xlim([0, 200])
        plt.gca().set_ylim([300,0])
        plt.gca().axes.get_xaxis().set_ticks([])
        plt.gca().axes.get_yaxis().set_ticks([])
        plt.title('Original Image',fontsize=12)

        p = 2
        for d in datasets:
            plt.subplot(2, n_panels, p)
            plt.imshow(frontImage[d], cmap='gray')
            plt.plot(pixels[d][:, 0], pixels[d][:, 1], 'y-',linewidth=3)
            plt.gca().set_xlim([0, 200])
            plt.gca().set_ylim([300, 0])
            plt.gca().axes.get_xaxis().set_ticks([])
            plt.gca().axes.get_yaxis().set_ticks([])
            plt.title('%s Solution'%d,fontsize=12)
            p += 1


        plt.subplot(2,n_panels,p)
        plt.title('Geocoded Solutions',fontsize=12)
        plt.ylabel('Northing (km)',fontsize=12)
        plt.xlabel('Easting (km)',fontsize=12)
        plt.plot(trueFront[:,0]/1000,trueFront[:,1]/1000,'k-',label='True')
        for d,c in zip(datasets,['b-','g-','r-']):
            plt.plot(front[d][:,0]/1000,front[d][:,1]/1000,c,label=d)
        plt.gca().set_xlim([frontXmin/1000,frontXmax/1000])
        plt.gca().set_ylim([frontYmin/1000, frontYmax/1000])
        plt.gca().set_xticks([frontXmin/1000,frontXmax/1000])
        plt.gca().set_yticks([frontYmin / 1000, frontYmax / 1000])
        plt.legend(loc=0)

        p += 1
        p_temp = copy.copy(p)
        x = {}
        y = {}
        for d,c in zip(datasets,['b','g','r']):
            plt.subplot(2,n_panels,p)
            plt.title('%s Errors Histogram'%d,fontsize=12)
            bins=range(0,5000,100)
            y[d], x[d], _ =plt.hist(errors[d],alpha=0.5,color=c,bins=bins,label='NN')
            plt.xlabel('RMS Error = '+'{0:.2f}'.format(rmsError(errors[d]))+' m',fontsize=12)

            p += 1

        #-- set histogram bounds
        for d in datasets:
            plt.subplot(2,n_panels,p_temp)
            plt.gca().set_ylim([0,np.max([y[d] for d in datasets])])
            plt.gca().set_xlim([0, np.max([x[d] for d in datasets])])
            p_temp += 1

        plt.savefig(os.path.join(outputFolder, label + '.png'),bbox_inches='tight')
        plt.close(fig)

    fig=plt.figure(figsize=(11,4))

    x = {}
    y = {}
    for i,d,c in zip(range(len(datasets)),datasets,['b','g','r']):
        plt.subplot(1,len(datasets),i+1)
        plt.title(r"$\bf{e)}$" + " %s Error Histogram"%d,fontsize=12)
        bins=range(0,5000,100)
        y[d], x[d], _ =plt.hist(allerrors[d],alpha=0.5,color=c,bins=bins,label=d)
        plt.xlabel('RMS Error = '+'{0:.2f}'.format(rmsError(allerrors[d]))+' m',fontsize=12)
        if i==0:
            plt.ylabel('Count (100 m bins)',fontsize=12)

    for i in range(len(datasets)):
        plt.subplot(1,len(datasets),i+1)
        plt.gca().set_ylim([0,np.max([y[d] for d in datasets])])
        plt.gca().set_xlim([0,np.max([x[d] for d in datasets])])

    plt.savefig(os.path.join(results_dir,\
        'Figure_4_'+'_'.join(method.split())+'_'+str(step)+'_%isegs'%n_interval+'.pdf'),bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()