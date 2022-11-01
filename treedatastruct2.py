import sys
import csv
import os
from datetime import datetime
from playsound import playsound
import pandas as pd
import numpy as np
import seaborn as sn 
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from scipy.stats import pearsonr
from scipy.stats import f_oneway

class BinarySearchTreeNode:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None
    def add_child(self, data):
        if data == self.data:
            return # node already exist

        if data < self.data:
            if self.left:
                self.left.add_child(data)
            else:
                self.left = BinarySearchTreeNode(data)
        else:
            if self.right:
                self.right.add_child(data)
            else:
                self.right = BinarySearchTreeNode(data)
    def search(self, val):
        if self.data == val:
            return True

        if val < self.data:
            if self.left:
                return self.left.search(val)
            else:
                return False

        if val > self.data:
            if self.right:
                return self.right.search(val)
            else:
                return False
    def in_order_traversal(self):
        elements = []
        if self.left:
            elements += self.left.in_order_traversal()

        elements.append(self.data)

        if self.right:
            elements += self.right.in_order_traversal()

        return elements
    def delete(self, val):
        if val < self.data:
            if self.left:
                self.left = self.left.delete(val)
        elif val > self.data:
            if self.right:
                self.right = self.right.delete(val)
        else:
            if self.left is None and self.right is None:
                return None
            elif self.left is None:
                return self.right
            elif self.right is None:
                return self.left

            min_val = self.right.find_min()
            self.data = min_val
            self.right = self.right.delete(min_val)
        return self
    def find_max(self):
        if self.right is None:
            return self.data
        return self.right.find_max()
    def find_min(self):
        if self.left is None:
            return self.data
        return self.left.find_min()

class OrdinaryLeastSquares(object):

  def _init_(self):
    self.coefficients = []

  def fit(self, X, y):
    if len(X.shape) == 1: X = self._reshape_x(X)

    X = self._concatenate_ones(X)
    self.coefficients = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(y)

  def predict(self, entry):
    b0 = self.coefficients[0]
    other_betas = self.coefficients[1:]
    prediction = b0

    for xi, bi in zip(entry, other_betas): prediction += (bi *xi)
    return prediction

  def _reshape_x(self, X):
    return X.reshape(-1, 1)

  def _concatenate_ones(self, X):
    ones = np.ones(shape=X.shape[0]).reshape(-1, 1)
    return np.concatenate((ones, X), 1)

def insertion_sort(elements):
    for i in range(1, len(elements)):
        anchor = elements[i]
        j = i - 1
        while j>=0 and anchor < elements[j]:
            elements[j+1] = elements[j]
            j = j - 1
        elements[j+1] = anchor
    
    return elements
   
def build_tree(elements):
    root1 = BinarySearchTreeNode(elements[0]) 
    for i in range(1,len(elements)):
        root1.add_child(elements[i])
    return root1

def build_tree1(elements1):

    root2 = BinarySearchTreeNode(elements1[0])
    
    for i in range(1,len(elements1)):
        root2.add_child(elements1[i])

    return root2

playsound("britannia.mp3")
print("Analysis Super Genie")
name = input("Enter Your Name: ")
now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
timee = name+" "+dt_string
remember = open("Login.txt","a")
remember.write("\n")
remember.write(timee)
remember.close()

floc = input("Kindly enter the file location: ")
fname = input("Enter file name: ")
file = floc+fname+'.csv'
df = pd.read_csv(file)

reporttxt = fname+".txt"
report = open(reporttxt,"a")
report.write("Data Set Name: ")
report.write(fname)
report.write("\n")
report.write("Application Usage Time: ")
report.write(dt_string)
report.write(" User: ")
report.write(name)
report.close()

def defult():
    print(" 1.Confirm DataSet for Analysis \n 2.Search Data \n 3.Admin \n 4.Exit")
    ch = int(input("Enter your choice: "))
    if(ch == 1):
        print(df.head())
        column_name = list(df.columns)
        print("\nAvailable Columns in the Dataset: \n")
        listToStr = '\n'.join([str(elem) for elem in column_name])
        print(listToStr)
        print("")
        choice = input("Proceed with the dataset? (Y/N)")
        if(choice == 'N'):
            print("Reload Genie Again!")
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            remember = open("Login.txt","a")
            remember.write(" ")
            remember.write(dt_string)
            remember.close()
            sys.exit()
        else:
            analysis()
    elif(ch == 2):
            dsa()
    elif(ch == 3):
            admin()
    elif(ch == 4):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        remember = open("Login.txt","a")
        remember.write(" ")
        remember.write(dt_string)
        remember.close()
        report = open(reporttxt,"a")
        report.write("\n")
        report.write("END OF REPORT!")
        report.close()
        print(fname+" file created successfully! You can find it in the file explorer")
        sys.exit()
    else:
        print("Enter Proper Navigation Code")
        defult()

   

def analysis():
        print(" 1.Check Null Values \n 2.Data Description \n 3.Visualisations \n 4.Heat Map \n 5.Statistical Tests \n 6.Histogram \n 7.Back")
        ch = int(input("Enter your choice: "))
        if(ch == 7):
            defult()
        dep = input("Enter the Dependent Column Name: ")
        cat = input("Enter the Categorical Column Name: ")
        dep2 = input("Enter Second Dependent Column: ")
        df[dep] = df[dep].astype('int') 
        if(ch == 1):
            print("-----------------------Checking for Null Values------------------------")
            print(df.isnull())
            report = open(reporttxt,"a")
            report.write("\n")
            report.write("Null values in data checked!")
            report.close()
            analysis()
        elif(ch == 2):
            print("---------------------------Data Description----------------------------")
            print(df.describe().T)
            report = open(reporttxt,"a")
            report.write("\n")
            report.write("Data is described to user!")
            report.close()
            analysis()
        elif(ch == 4):
            print("-----------------------------HEAT MAP----------------------------------")
            corr = df.corr(method = 'spearman')
            print(corr)
            corr = df.corr(method = 'spearman')
            plt.figure(figsize = (6,6))
            sn.heatmap(corr, linewidths= 0.1 , square = True, annot = True, cmap= 'GnBu',linecolor='black')
            report = open(reporttxt,"a")
            report.write("\n")
            report.write("Heat Map visualised")
            report.close()
            analysis()
        elif(ch == 3):
            print("--------------------------VISUALISATIONS-------------------------------")
            print(" 1.PieChart \n 2.HeatMap")
            daata = df.groupby(cat)[dep].sum()
            daata.plot.pie(autopct = "%.1f%%") 
            cor = df.corr().iloc[ : , -1].sort_values(ascending=False)[1:].to_frame()
            sn.set(font_scale=1.6)
            plt.figure(figsize=(12, 8))
            sn.heatmap(data= cor, cmap="jet",center=0.1, annot=True, vmax=.5, linewidths=0.1,annot_kws={"size": 16})
            plt.show()
            report = open(reporttxt,"a")
            report.write("\n")
            report.write("Pie Chart and regression plotted")
            report.close()
            analysis()
        elif(ch == 5):
            print(" 1.Normality Test \n 2.Correlation Test \n 3.Hypothesis Test \n 4.Back")
            ch = int(input("Enter your choice: "))
            if(ch == 1):
                print("--------------------------------------------------------")
                print("                      Shapiro Test                      ")
                nor_test = df[dep].tolist()
                stat, p = shapiro(nor_test)
                print('stat=%.3f, p=%.3f' % (stat, p))
                if (p > 0.05):
                    	print('Probably Gaussian')
                else:
	                    print('Probably not Gaussian')
                print("--------------------------------------------------------")
            elif(ch == 2):
                print("--------------------------------------------------------")
                print("              Pearson Correlation Coefficient           ")
                per_test1 = df[dep].tolist()
                per_test2 = df[dep2].tolist()
                stat, p = pearsonr(per_test1, per_test2)
                print('stat=%.3f, p=%.3f' % (stat, p))
                if p > 0.05:
                	print('Probably independent')
                else:
                	print('Probably dependent')
                print("--------------------------------------------------------")
            elif(ch == 3):
                print("--------------------------------------------------------")
                print("                  Analysis of Variance Test             ")
                stat, p = f_oneway(dep, dep2)
                print('stat=%.3f, p=%.3f' % (stat, p))
                if p > 0.05:
                	print('Probably the same distribution')
                else:
                	print('Probably different distributions')
            else:
                analysis()
            report = open(reporttxt,"a")
            report.write("\n")
            report.write("Statistical tests performed and data set is made clear")
            report.close()
            analysis()
            
        elif(ch == 6):
            sn.distplot( a=df[dep], hist=True, kde=False, rug=False )
            plt.show()
            analysis()
def dsa():
    avail_column = list(df.columns)
    column_tree = build_tree(avail_column)
    print("\n 1.Search Columns \n 2.Sort Column List \n 3.Search Data \n 4.Sort Data \n 5.Add Data into CSV \n 6.Back")
    choice = int(input("Enter your choice: "))
    if(choice == 1):
        data = input("Enter Column Name: ")
        res = column_tree.search(data)
        if res == True:
            print("Column Present")
            dsa()
        else:
            print("Column not present")
            dsa()
    elif(choice == 2):
        print("Sorting the Whole tree using Insertion Sort")
        col_sort=insertion_sort(avail_column)
        listToStr1 = '\n'.join([str(elem) for elem in col_sort])
        print(listToStr1)
        print("\n")
        dsa()
    elif(choice == 3):
        print("Available Columns:")
        col_sort=insertion_sort(avail_column)
        listToStr1 = '\n'.join([str(elem) for elem in col_sort])
        print(listToStr1)
        print("\n")
        col_namee = input("Enter Column Name: ")
        col = df[col_namee].tolist()
        data_treee = build_tree(col)
        dataa = input("Enter Value: ")
        while True:
            rees = data_treee.search(dataa)
            try:
                if rees == True:
                    print("Data Present")
                    break
            except ValueError:
                print("Data not found")
        dsa()
    elif(choice == 4):
        col_namee = input("Enter Column Name: ")
        df[col_namee] = df[col_namee].astype('int')
        col = df[col_namee].tolist()
        data_treee = build_tree(col)
        print("Sorting the Whole tree using Insertion Sort")
        col_sort=insertion_sort(col)
        listToStr1 = '\n'.join([str(elem) for elem in col_sort])
        print(listToStr1)
        print("\n")
        dsa()
    elif(choice == 5):
        print("Enter Data in the following Format")
        print(avail_column)
        darta = input()
        with open(file, 'a') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(darta)
        print("Data Saved")
        dsa()
    elif(choice == 6):
        defult()
    else:
        print("Enter Correct Navigation ID")
        dsa()
    report = open(reporttxt,"a")
    report.write("\n")
    report.write("Usage of Data Structures implemented and data is been sorted!")
    report.close()   
def admin():
    print("ADMIN")
    with open('Login.txt') as f:
        contents = f.read()
        print(contents)
while True:
    os.system('cls')
    defult()
