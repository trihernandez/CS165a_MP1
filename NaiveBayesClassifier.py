import numpy
import scipy
#import pandas
import hashlib
import fileinput
import timeit #to calculate runtime

import sys

#Naive Bayes:
#P(X_1,...X_n|C) = PI(i=1,n)(X_i|C)

#Updates the average by adding a new element to a set given the number of elements and average
def update_avg(NumElements, OldAvg, NewElement):
    Numerator = (NumElements * OldAvg) + NewElement
    Denominator = NumElements + 1
    return Numerator / Denominator

#Algorithm for else case taken from:
#https://math.stackexchange.com/questions/775391/can-i-calculate-the-new-standard-deviation-when-adding-a-value-without-knowing-t
#Updates the variance by adding a new element to a set given the number of elements, average, and variance
#Also updates the average and returns both the updated variance and average
def update_avg_and_var(NumElements, OldAvg, OldVar, NewElement):
    NewAvg = update_avg(NumElements, OldAvg, NewElement)
    
    #if there is only one element in the existing array just get the distance from the middle of the two variables
    #else use the formula denoted from the above site
    if(NumElements==1):
        NewVar = ( (OldAvg - NewAvg)**2 ) + ( (NewElement - NewAvg)**2 )
    else:
        Numerator_var = ( (NumElements-1) * OldVar ) + ( (NewElement-NewAvg) * (NewElement-OldAvg) )
        NewVar = Numerator_var / NumElements
    return NewAvg, NewVar

#Changes directions into to_degrees
#Starts at east and continues counter-clockwise
def to_degrees(Direction):
    if (Direction=="E"): return 0.0
    elif (Direction=="ENE"): return 22.5
    elif (Direction=="NE"):  return 45.0
    elif (Direction=="NNE"): return 67.5
    
    elif (Direction=="N"):   return 90.0
    elif (Direction=="NNW"): return 112.5
    elif (Direction=="NW"):  return 135.0
    elif (Direction=="WNW"): return 157.5
    
    elif (Direction=="W"):   return 180.0
    elif (Direction=="WSW"): return 202.5
    elif (Direction=="SW"):  return 225.0
    elif (Direction=="SSW"): return 247.5
    
    elif (Direction=="S"):   return 270.0
    elif (Direction=="SSE"): return 292.5
    elif (Direction=="SE"):  return 315.0
    elif (Direction=="ESE"): return 337.5
    else:
        print("========================")
        print("Direction not recognized")
        print("========================")
        return 9000.1

def get_x_and_y_components(Magnitude, Direction):
    x = Magnitude * numpy.cos( Direction * numpy.pi/180 )
    y = Magnitude * numpy.sin( Direction * numpy.pi/180 )
    return x,y

#Add an x and y vector to the list given
def update_avg_and_var_vector(NumElements, OldAvg_x, OldAvg_y, OldVar_x, OldVar_y, NewMagnitude, NewDirection):
    NewElement_x, NewElement_y = get_x_and_y_components(NewMagnitude, NewDirection)

    NewAvg_x,NewVar_x = update_avg_and_var(NumElements, OldAvg_x, OldVar_x, NewElement_x)
    NewAvg_y,NewVar_y = update_avg_and_var(NumElements, OldAvg_y, OldVar_y, NewElement_y)
    return NewAvg_x, NewAvg_y, NewVar_x, NewVar_y
  
#Return a normal distribution probability of given mean, st.dev, and element
#multiply by the st.dev to adjust the value on the curve so it's the same based on the number of st.dev from the mean
#and the mean for this distribution returns 1
def normalized_probability(avg, stdev, element):
    LeftPart = 1 / ( stdev * numpy.sqrt(2 * numpy.pi) )
    Exponent = -(element-avg)**2 / ( 2*stdev**2 )
    RightPart = 2.71828182845904523536**Exponent
    return LeftPart * RightPart * stdev/(0.3989422804014327)










class YesNoData:
    def __init__ (self, Location, MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine, WindGustDir, WindGustSpeed, WindDir9am, WindDir3pm, WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am, Pressure3pm, Cloud9am, Cloud3pm, Temp9am, Temp3pm, RainToday, RainTomorrow):
        #name of this location
        self.RainTomorrow = RainTomorrow
        
        self.Location = Location
        
        #number of cases
        self.NumCases = 1
        #avg min/max temperatures given that it rains/doesn't rain tomorrow
        self.MinTemp_avg = float(MinTemp)
        self.MaxTemp_avg = float(MaxTemp)
        self.MinTemp_stdev = float(0)
        self.MaxTemp_stdev = float(0)
        #avg rainfall given that it rains/doesn't rain tomorrow
        self.Rainfall_avg = float(Rainfall)
        self.Rainfall_stdev = float(0)
        #avg direction given that it rains/doesn't rain tomorrow
        self.Evaporation_avg = float(Evaporation)
        self.Evaporation_stdev = float(0)
        #avg sunshine given that it rains/doesn't rain tomorrow
        self.Sunshine_avg = float(Sunshine)
        self.Sunshine_stdev = float(0)
        #avg wind gust x and y magnitudes given that it rains/doesn't rain tomorrow
        self.WindGustX_avg, self.WindGustY_avg = get_x_and_y_components(float(WindGustSpeed), to_degrees(WindGustDir) )
        self.WindGustX_stdev = float(0)
        self.WindGustY_stdev = float(0)
        #avg 9am and 3pm wind x and y magnitudes given that it rains/doesn't rain tomorrow
        self.Wind9amX_avg, self.Wind9amY_avg = get_x_and_y_components(float(WindSpeed9am), to_degrees(WindDir9am) )
        self.Wind9amX_stdev = float(0)
        self.Wind9amY_stdev = float(0)
        self.Wind3pmX_avg, self.Wind3pmY_avg = get_x_and_y_components(float(WindSpeed3pm), to_degrees(WindDir3pm) )
        self.Wind3pmX_stdev = float(0)
        self.Wind3pmY_stdev = float(0)
        #avg 9am/3pm humidity given that it rains/doesn't rain tomorrow
        self.Humidity9am_avg = float(Humidity9am)
        self.Humidity3pm_avg = float(Humidity3pm)
        self.Humidity9am_stdev = float(0)
        self.Humidity3pm_stdev = float(0)
        #avg 9am/3pm air pressure given that it rains/doesn't rain tomorrow
        self.Pressure9am_avg = float(Pressure9am)
        self.Pressure3pm_avg = float(Pressure3pm)
        self.Pressure9am_stdev = float(0)
        self.Pressure3pm_stdev = float(0)
        #avg 9am/3pm cloudiness given that it rains/doesn't rain tomorrow
        self.Cloud9am_avg = float(Cloud9am)
        self.Cloud3pm_avg = float(Cloud3pm)
        self.Cloud9am_stdev = float(0)
        self.Cloud3pm_stdev = float(0)
        #avg 9am/3pm temperature given that it rains/doesn't rain tomorrow
        self.Temp9am_avg = float(Temp9am)
        self.Temp3pm_avg = float(Temp3pm)
        self.Temp9am_stdev = float(0)
        self.Temp3pm_stdev = float(0)
        #percent chance that it rains today given that it rains/doesn't rain tomorrow
        if(RainToday == "Yes"):
            self.RainToday_percentage = 1.0
        if(RainToday == "No"):
            self.RainToday_percentage = float(0)
            
    def add_data (self, Location, MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine, WindGustDir, WindGustSpeed, WindDir9am, WindDir3pm, WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am, Pressure3pm, Cloud9am, Cloud3pm, Temp9am, Temp3pm, RainToday, RainTomorrow):
        
        #update min/max temperatures
        self.MinTemp_avg, self.MinTemp_stdev = update_avg_and_var(self.NumCases, self.MinTemp_avg, self.MinTemp_stdev**2, float(MinTemp))
        self.MinTemp_stdev = numpy.sqrt(self.MinTemp_stdev)
        self.MaxTemp_avg, self.MaxTemp_stdev = update_avg_and_var(self.NumCases, self.MaxTemp_avg, self.MaxTemp_stdev**2, float(MaxTemp))
        self.MaxTemp_stdev = numpy.sqrt(self.MaxTemp_stdev)
        
        #update rainfall averages, stdev
        self.Rainfall_avg, self.Rainfall_stdev = update_avg_and_var(self.NumCases, self.Rainfall_avg, self.Rainfall_stdev**2, float(Rainfall))
        self.Rainfall_stdev = numpy.sqrt(self.Rainfall_stdev)
        
        #update evaporation averages
        self.Evaporation_avg, self.Evaporation_stdev = update_avg_and_var(self.NumCases, self.Evaporation_avg, self.Evaporation_stdev**2, float(Evaporation))
        self.Evaporation_stdev = numpy.sqrt(self.Evaporation_stdev)
        
        #update sunshine
        self.Sunshine_avg, self.Sunshine_stdev = update_avg_and_var(self.NumCases, self.Sunshine_avg, self.Sunshine_stdev**2, float(Sunshine))
        self.Sunshine_stdev = numpy.sqrt(self.Sunshine_stdev)
        
        #update windgust
        self.WindGustX_avg, self.WindGustY_avg, self.WindGustX_stdev, self.WindGustY_stdev = update_avg_and_var_vector(self.NumCases, self.WindGustX_avg, self.WindGustY_avg, self.WindGustX_stdev**2, self.WindGustY_stdev**2, float(WindGustSpeed), to_degrees(WindGustDir) )
        self.WindGustX_stdev = numpy.sqrt(self.WindGustX_stdev)
        self.WindGustY_stdev = numpy.sqrt(self.WindGustY_stdev)

        #update wind at 9am and 3pm
        self.Wind9amX_avg, self.Wind9amY_avg, self.Wind9amX_stdev, self.Wind9amY_stdev = update_avg_and_var_vector(self.NumCases, self.Wind9amX_avg, self.Wind9amY_avg, self.Wind9amX_stdev**2, self.Wind9amY_stdev**2, float(WindSpeed9am), to_degrees(WindDir9am) )
        self.Wind3pmX_avg, self.Wind3pmY_avg, self.Wind3pmX_stdev, self.Wind3pmY_stdev = update_avg_and_var_vector(self.NumCases, self.Wind3pmX_avg, self.Wind3pmY_avg, self.Wind3pmX_stdev**2, self.Wind3pmY_stdev**2, float(WindSpeed3pm), to_degrees(WindDir3pm) )
        self.Wind9amX_stdev = numpy.sqrt(self.Wind9amX_stdev)
        self.Wind9amY_stdev = numpy.sqrt(self.Wind9amY_stdev)
        self.Wind3pmX_stdev = numpy.sqrt(self.Wind3pmX_stdev)
        self.Wind3pmY_stdev = numpy.sqrt(self.Wind3pmY_stdev)

        #update humidities
        self.Humidity9am_avg, self.Humidity9am_stdev = update_avg_and_var(self.NumCases, self.Humidity9am_avg, self.Humidity9am_stdev**2, float(Humidity9am))
        self.Humidity3pm_avg, self.Humidity3pm_stdev = update_avg_and_var(self.NumCases, self.Humidity3pm_avg, self.Humidity3pm_stdev**2, float(Humidity3pm))
        self.Humidity9am_stdev = numpy.sqrt(self.Humidity9am_stdev)
        self.Humidity3pm_stdev = numpy.sqrt(self.Humidity3pm_stdev)
        
        #update pressures
        self.Pressure9am_avg, self.Pressure9am_stdev = update_avg_and_var(self.NumCases, self.Pressure9am_avg, self.Pressure9am_stdev**2, float(Pressure9am))
        self.Pressure3pm_avg, self.Pressure3pm_stdev = update_avg_and_var(self.NumCases, self.Pressure3pm_avg, self.Pressure3pm_stdev**2, float(Pressure3pm))
        self.Pressure9am_stdev = numpy.sqrt(self.Pressure9am_stdev)
        self.Pressure3pm_stdev = numpy.sqrt(self.Pressure3pm_stdev)
        
        #update cloudiness
        self.Cloud9am_avg, self.Cloud9am_stdev = update_avg_and_var(self.NumCases, self.Cloud9am_avg, self.Cloud9am_stdev**2, float(Cloud9am))
        self.Cloud3pm_avg, self.Cloud3pm_stdev = update_avg_and_var(self.NumCases, self.Cloud3pm_avg, self.Cloud3pm_stdev**2, float(Cloud3pm))
        self.Cloud9am_stdev = numpy.sqrt(self.Cloud9am_stdev)
        self.Cloud3pm_stdev = numpy.sqrt(self.Cloud3pm_stdev)
        
        #update temperatures
        self.Temp9am_avg, self.Temp9am_stdev = update_avg_and_var(self.NumCases, self.Temp9am_avg, self.Temp9am_stdev**2, float(Temp9am))
        self.Temp3pm_avg, self.Temp3pm_stdev = update_avg_and_var(self.NumCases, self.Temp3pm_avg, self.Temp3pm_stdev**2, float(Temp3pm))
        self.Temp9am_stdev = numpy.sqrt(self.Temp9am_stdev)
        self.Temp3pm_stdev = numpy.sqrt(self.Temp3pm_stdev)
        
        #update raintoday
        if(RainToday == "Yes"):
            self.RainToday_percentage = update_avg(self.NumCases, self.RainToday_percentage, 1.0)
        if(RainToday == "No"):
            self.RainToday_percentage = update_avg(self.NumCases, self.RainToday_percentage, 0.0)

        #The case has been added, update the number of cases
        self.NumCases = self.NumCases + 1
    
    #Multiply the bayesian products for all numeric variables 
    def naive_bayes(self, MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine, WindGustDir, WindGustSpeed, WindDir9am, WindDir3pm, WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am, Pressure3pm, Cloud9am, Cloud3pm, Temp9am, Temp3pm, RainToday):
        #get x and y magnitudes for WindGust, Wind9am, and Wind3pm
        X_gust, Y_gust = get_x_and_y_components( float(WindGustSpeed), to_degrees(WindGustDir) )
        X_9am, Y_9am = get_x_and_y_components( float(WindSpeed9am), to_degrees(WindDir9am) )
        X_3pm, Y_3pm = get_x_and_y_components( float(WindSpeed3pm), to_degrees(WindDir3pm) )
        
        #get bayesian for cases of yes/no for location / total cases of yes/no

        #get all bayesian products
        b01 = normalized_probability(self.MinTemp_avg, self.MinTemp_stdev, float(MinTemp))
        b02 = normalized_probability(self.MaxTemp_avg, self.MaxTemp_stdev, float(MaxTemp))
        b03 = normalized_probability(self.Rainfall_avg, self.Rainfall_stdev, float(Rainfall))
        b04 = normalized_probability(self.Evaporation_avg, self.Evaporation_stdev, float(Evaporation))
        b05 = normalized_probability(self.Sunshine_avg, self.Sunshine_stdev, float(Sunshine))
        
        b06 = normalized_probability(self.WindGustX_avg, self.WindGustX_stdev, float(X_gust))
        b07 = normalized_probability(self.WindGustY_avg, self.WindGustY_stdev, float(Y_gust))
        
        b08 = normalized_probability(self.Wind9amX_avg, self.Wind9amX_stdev, float(X_9am))
        b09 = normalized_probability(self.Wind9amY_avg, self.Wind9amY_stdev, float(Y_9am))
        b10 = normalized_probability(self.Wind3pmX_avg, self.Wind3pmX_stdev, float(X_3pm))
        b11 = normalized_probability(self.Wind3pmY_avg, self.Wind3pmY_stdev, float(Y_3pm))
        
        b12 = normalized_probability(self.Humidity9am_avg, self.Humidity9am_stdev, float(Humidity9am))
        b13 = normalized_probability(self.Humidity3pm_avg, self.Humidity3pm_stdev, float(Humidity3pm))
        b14 = normalized_probability(self.Pressure9am_avg, self.Pressure9am_stdev, float(Pressure9am))
        b15 = normalized_probability(self.Pressure3pm_avg, self.Pressure3pm_stdev, float(Pressure3pm))
        b16 = normalized_probability(self.Cloud9am_avg, self.Cloud9am_stdev, float(Cloud9am))
        b17 = normalized_probability(self.Cloud3pm_avg, self.Cloud3pm_stdev, float(Cloud3pm))
        b18 = normalized_probability(self.Temp9am_avg, self.Temp9am_stdev, float(Temp9am))
        b19 = normalized_probability(self.Temp3pm_avg, self.Temp3pm_stdev, float(Temp3pm))

        return b01 * b02 * b03 * b04 * b05 * b06 * b07 * b08 * b09 * b10 * b11 * b12 * b13 * b14 * b15 * b16 * b17 * b18 * b19











class LocationData:
    def __init__ (self, Location, MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine, WindGustDir, WindGustSpeed, WindDir9am, WindDir3pm, WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am, Pressure3pm, Cloud9am, Cloud3pm, Temp9am, Temp3pm, RainToday, RainTomorrow):
        self.Location = Location
        if(RainTomorrow == "Yes"):
            self.yDat = YesNoData(Location, MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine, WindGustDir, WindGustSpeed, WindDir9am, WindDir3pm, WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am, Pressure3pm, Cloud9am, Cloud3pm, Temp9am, Temp3pm, RainToday, RainTomorrow)
            self.nDat = None
        if(RainTomorrow == "No"):
            self.yDat = None
            self.nDat = YesNoData(Location, MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine, WindGustDir, WindGustSpeed, WindDir9am, WindDir3pm, WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am, Pressure3pm, Cloud9am, Cloud3pm, Temp9am, Temp3pm, RainToday, RainTomorrow)
        
    def add_data(self, Location, MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine, WindGustDir, WindGustSpeed, WindDir9am, WindDir3pm, WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am, Pressure3pm, Cloud9am, Cloud3pm, Temp9am, Temp3pm, RainToday, RainTomorrow):
        if(RainTomorrow == "Yes"):
            if(self.yDat == None):
                self.yDat = YesNoData(Location, MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine, WindGustDir, WindGustSpeed, WindDir9am, WindDir3pm, WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am, Pressure3pm, Cloud9am, Cloud3pm, Temp9am, Temp3pm, RainToday, RainTomorrow)
            else:
                self.yDat.add_data(Location, MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine, WindGustDir, WindGustSpeed, WindDir9am, WindDir3pm, WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am, Pressure3pm, Cloud9am, Cloud3pm, Temp9am, Temp3pm, RainToday, RainTomorrow)
        else:
            if(self.nDat == None):
                self.nDat = YesNoData(Location, MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine, WindGustDir, WindGustSpeed, WindDir9am, WindDir3pm, WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am, Pressure3pm, Cloud9am, Cloud3pm, Temp9am, Temp3pm, RainToday, RainTomorrow)
            else:
                self.nDat.add_data(Location, MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine, WindGustDir, WindGustSpeed, WindDir9am, WindDir3pm, WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am, Pressure3pm, Cloud9am, Cloud3pm, Temp9am, Temp3pm, RainToday, RainTomorrow)
        
    def naive_bayes(self, TotalYes, TotalNo, MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine, WindGustDir, WindGustSpeed, WindDir9am, WindDir3pm, WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am, Pressure3pm, Cloud9am, Cloud3pm, Temp9am, Temp3pm, RainToday):
        Bayes_yes = self.yDat.naive_bayes(MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine, WindGustDir, WindGustSpeed, WindDir9am, WindDir3pm, WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am, Pressure3pm, Cloud9am, Cloud3pm, Temp9am, Temp3pm, RainToday)
        Bayes_no = self.nDat.naive_bayes(MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine, WindGustDir, WindGustSpeed, WindDir9am, WindDir3pm, WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am, Pressure3pm, Cloud9am, Cloud3pm, Temp9am, Temp3pm, RainToday)
        #prevents the learner from being indecisive if the two binary predictors are too low.
        if(RainToday == yes):
            if (self.nDat.RainToday_percentage < 10**-10 and self.yDat.RainToday_percentage < 10**-10):
                Bayes_yes = Bayes_yes
                Bayes_no = Bayes_no
            else:
                Bayes_yes = Bayes_yes * self.yDat.RainToday_percentage
                Bayes_no = Bayes_no * self.nDat.RainToday_percentage
        else:
            NoRainToday_yDat = 1 - self.yDat.RainToday_percentage
            NoRainToday_nDat = 1 - self.nDat.RainToday_percentage
            if (NoRainToday_yDat < 10**-10 and NoRainToday_nDat < 10**-10):
                Bayes_yes = Bayes_yes
                Bayes_no = Bayes_no
            else:
                Bayes_yes = Bayes_yes * NoRainToday_yDat
                Bayes_no = Bayes_no * NoRainToday_nDat
            
        #return the larger probability
        if(Bayes_yes > Bayes_no ):
            return 1
        else:
            return 0










def main():
    #Start our Program
    
    start = timeit.default_timer()

    #input training data
    DataMap = {}
    TotalData = 0
    TotalYes = 0
    TotalNo = 0
    
    TotalTests = 0
    TotalRight = 0
    
    for line in fileinput.input(files ='training.txt'):

        SplitInput = line.split("\n")
        SplitInput = SplitInput[0].split(", ")

        if(SplitInput[0] in DataMap):
            DataMap[ SplitInput[0] ].add_data(SplitInput[0],SplitInput[1],SplitInput[2],SplitInput[3],SplitInput[4],SplitInput[5],SplitInput[6],SplitInput[7],SplitInput[8],SplitInput[9],SplitInput[10],SplitInput[11],SplitInput[12],SplitInput[13],SplitInput[14],SplitInput[15],SplitInput[16],SplitInput[17],SplitInput[18],SplitInput[19],SplitInput[20],SplitInput[21])
        else:
            NewData = LocationData(SplitInput[0],SplitInput[1],SplitInput[2],SplitInput[3],SplitInput[4],SplitInput[5],SplitInput[6],SplitInput[7],SplitInput[8],SplitInput[9],SplitInput[10],SplitInput[11],SplitInput[12],SplitInput[13],SplitInput[14],SplitInput[15],SplitInput[16],SplitInput[17],SplitInput[18],SplitInput[19],SplitInput[20],SplitInput[21])
            DataMap[ SplitInput[0] ] = NewData
        
        TotalData = TotalData + 1
        if(SplitInput[21] == "Yes"):
            TotalYes = TotalYes + 1
        if(SplitInput[21] == "No"):
            TotalNo = TotalNo + 1
    
    #Start reading our test data
    for line in fileinput.input(files ='private_testing_input.txt'):
        SplitInput = line.split("\n")
        SplitInput = SplitInput[0].split(", ")
        BayesPrediction = DataMap[ SplitInput[0] ].naive_bayes(TotalYes,TotalNo,SplitInput[1],SplitInput[2],SplitInput[3],SplitInput[4],SplitInput[5],SplitInput[6],SplitInput[7],SplitInput[8],SplitInput[9],SplitInput[10],SplitInput[11],SplitInput[12],SplitInput[13],SplitInput[14],SplitInput[15],SplitInput[16],SplitInput[17],SplitInput[18],SplitInput[19],SplitInput[20])
        print(BayesPrediction)
        if(BayesPrediction == 1 and SplitInput[21] == "Yes"):
            TotalRight = TotalRight + 1
        if(BayesPrediction == 0 and SplitInput[21] == "No"):
            TotalRight = TotalRight + 1
        TotalTests = TotalTests + 1

    end = timeit.default_timer()
    #print("Total Runtime: ", end - start)
    #print()
    #print("Accuracy: ", TotalRight,"/",TotalTests)
    #print("          ", 100 * TotalRight/TotalTests, "%")


if __name__=="__main__":
    main()
