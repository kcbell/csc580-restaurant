# -*- coding: cp932 -*-
#------------------------------#
#function   : initMax          #
#author     : Toshihiro Kuboi  #
#date       : Oct. 19, 2012    #
#arguments  : list of authors  #
#returns    : 2D-list          #
#description: initializes 2D-  #
#             array for confu- #
#             sion matrix.     #
#------------------------------#
def initMatrix(author_list):
    authors = sorted(author_list)  # sort in alphabetical order
    confusionMatrix = []
    for i in range(len(authors)+1):#init with 0
        confusionMatrix.append([0] * (len(authors) + 1))

    for i in range(len(authors)+1):#the first row is authors
        if i > 0:
            confusionMatrix[0][i]=authors[i-1]
        else:
            confusionMatrix[0][i]='  '
            
    for i in range(len(authors)+1):#the first column is authors
        if i > 0:
            confusionMatrix[i][0]=authors[i-1]
            
    return confusionMatrix

#--------------------------------#
#function   : keepScore          #
#author     : Toshihiro Kuboi    #
#date       : Oct. 19, 2012      #
#arguments  : predicted author   #
#           : actual author      #
#           : 2D-list for matrix #
#description: keeps the score of #
#             prediction.        #
#--------------------------------#
def keepScore(predicted,actual,matrix):
    idx1 = matrix[0].index(actual)
    idx2 = matrix[0].index(predicted)
    matrix[idx1][idx2] += 1


#--------------------------------#
#function   : drawMatrix         #
#author     : Toshihiro Kuboi    #
#date       : Oct. 19, 2012      #
#arguments  : 2D-list for matrix #
#           : number of col / row#
#description: draw the confusion #
#             matrix.            #
#--------------------------------#    
def drawMatrix(matrix, column):
    print '----------------'
    print 'confusion matrix'
    print '----------------'
    print '   -> predicted authors *names are converted to initials'
    totalCol = len(matrix[0])  #total number of authors (columns)

    frm = 0
    while totalCol > 0 :       #until all columns are printed
        if totalCol > column:  #if all columns do not fit in the screen
            col = column
        else:                  #otherwise, print all columns at once
            col = totalCol
            
        for i in range(len(matrix[0])):
            line = ''
            for j in range(col):
                if i == 0 and j != 0:   #column header
                    name = matrix[i][frm + j]
                    line = line + ('%4s' % (name[0] + name[name.index(' ')+1])) + '|'
                elif i != 0 and j == 0: #row header
                    name = matrix[i][j]
                    line = line + ('%4s' % (name[0] + name[name.index(' ')+1])) + '|'
                elif i == 0 and j == 0: #top left corner
                    line = line + ('%4s' % matrix[i][j]) + '|'
                else:
                    if matrix[i][frm + j] > 0:
                        if i != j:
                            sumCount = sumRow(matrix[i])
                            line = line + ('%.2f' % (1 - (matrix[i][frm + j] / sumCount))) + '|'
                        else:
                            sumCount = sumRow(matrix[i])
                            line = line + ('%.2f' % (1 - (matrix[i][frm + j] / sumCount))) + '|'
                    else:
                        line = line + ('%.2f' % (1 - matrix[i][frm + j])) + '|'
            print line
        totalCol -= col        #remaining columns to be printed
        frm = col


#--------------------------------#
#function   : sumRow             #
#author     : Toshihiro Kuboi    #
#date       : Oct. 23, 2012      #
#arguments  : 2D-list for matrix #
#           : number of col / row#
#returns    : the sum of count in#
#             a row.             #
#description: sum the count in a #
#             row.               #
#--------------------------------#    
def sumRow(lst):
    result = 0
    idx = 1
    while idx < len(lst):
        result += lst[idx]
        idx += 1
    return result
