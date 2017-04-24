##Script create dataframe of US congressional districts and their data for migration into db. Output saves into csv.
##Author: Eleonore Fournier-Tombs
##Last update: 25 June 2015
##Note: Currently skips some entries requiring different parsing, see districttable.csv

##1. Open List of States and Number of Districts
DistrictNumbers <- read.csv("districtNumbers.csv")

##2. Declare and run CreateList Function
MainList <- data.frame(district = NULL, url = NULL)

CreateList <- function(state, code, number, dataframe = MainList) {
      i = 1
      NewList <- data.frame(district = NULL, url = NULL)
      if (number == 1) {
            url <- paste("https://en.wikipedia.org/w/api.php?action=query&titles=", state, "'s ", "at-large_congressional_district&prop=revisions&rvprop=content&format=json", sep="")
            district <- paste(code, "-", 1, sep="")
            Temp <- data.frame(district = district, url = url)
            NewList <- rbind(NewList, Temp)
      }
      else {
            while (i <= number) {
                  if (i == 1 | i == 21 | i == 31 | i == 41 | i == 51) {
                        url <- paste("https://en.wikipedia.org/w/api.php?action=query&titles=", state, "'s ", i, "st_congressional_district&prop=revisions&rvprop=content&format=json", sep="")
                  }
                  else if (i == 2 | i == 22 | i == 32 | i == 42 | i == 52 ) {
                        url <- paste("https://en.wikipedia.org/w/api.php?action=query&titles=", state, "'s ", i, "nd_congressional_district&prop=revisions&rvprop=content&format=json", sep="")
                  }
                  else if (i == 3 | i == 23 | i == 33 | i == 43 | i == 53) {
                        url <- paste("https://en.wikipedia.org/w/api.php?action=query&titles=", state, "'s ", i, "rd_congressional_district&prop=revisions&rvprop=content&format=json", sep="")   
                  }
                  else {
                        url <- paste("https://en.wikipedia.org/w/api.php?action=query&titles=", state, "'s ", i, "th_congressional_district&prop=revisions&rvprop=content&format=json", sep="")     
                  }
                  district <- paste(code, "-", i, sep="")
                  Temp <- data.frame(district = district, url = url)
                  NewList <- rbind(NewList, Temp)
                  i <- i+1
            }
      }
      MainList <- rbind(dataframe, NewList)
      return(MainList)
}

##3.  Loop through DistrictNumbers file to get list of District codes and links

for (i in 1:nrow(DistrictNumbers)) {
      NewNumber <- DistrictNumbers$number[i]
      NewState <- DistrictNumbers$state[i]
      NewCode <- DistrictNumbers$code[i]
      MainList <- CreateList(NewState, NewCode, NewNumber)
}

##4. Calls to Wikipedia API from List of Congressional Districts
library(rjson)
DistrictTable <- data.frame(matrix(0, ncol = 11))

info <- c("state", "district number", "party", "percent urban", "percent rural", "population", "median income", "percent white", "percent black", "percent asian", "percent native american")

colnames(DistrictTable) <- info

for (i in 1:nrow(MainList)) {
      tryCatch({  ##Catches and handles out-of-bounds errors
      url <- MainList$url[i]
      print(c("url", url)) ##
      url <- gsub(" ", "_", url)
      url <- gsub("'", "%27", url)
      suppressWarnings( ##Removes warnings for print statements at the end of the loop
            jsonRaw <- fromJSON(readLines(url))
      )
      jsonMin <- jsonRaw$query$pages[[1]]$revisions[[1]][3]
      jsonString <- toString(jsonMin[[1]])
      
      ##Extract first 1200 characters to save on processing time
      index <- seq(1, nchar(jsonString), 1200)
      mod <- sapply(index, function(x) substring(jsonString, x, x+1199))
      mod <- strsplit(mod, "Info") ##Extract Infobox
      mod <- mod[[1]][2]
      mod2 <- strsplit(mod, "\n")
      mod2 <- mod2[[1]]
      
      output <- matrix(0, ncol = 11, nrow = 0)
      
      parse <- function(j) {
            reg <- paste0("^.*", j, ".*$")
            throughput <- grep(reg, mod2, value = T)
            throughput <- strsplit(throughput, "=")
            output[j] <- throughput[[1]][2]
            }
      
      output <- sapply(info, parse) 
      
      DistrictTable[i,] <- output
      print(DistrictTable[i,]) ##Shows each line as it is inserted
      },
      error = function(e){})   ##Ignores out of bounds errors
}

##4. Prints dataframe into csv
      write.csv(DistrictTable, "districttable.csv")