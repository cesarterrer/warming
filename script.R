library(metaforest)
library(caret)
library(tidyverse)
library(parallel)
library(doParallel)
registerDoParallel(31) # Run in parallel -> 31 cores

df <- read.csv("Dataset of effect of warming on biomass-20201210.csv",stringsAsFactors = TRUE) %>%
  rename(yi.total="RR",vi.total="Variance..v.",
         yi.above="RR.1",vi.above="Variance..v..1",
         yi.below="RR.2",vi.below="Variance..v..2",
         Myc="Mycorrhizal.Type") %>% # We want as many data points as possible, so let's create a new effect size column
  mutate(yi=coalesce(yi.total,yi.above,yi.below), # If total biomass not available, use aboveground. If aboveground not available, use root biomass
         vi=coalesce(vi.total,vi.above,vi.below), # same with variances
         biomass.type=as.factor(if_else(is.na(yi.total) & !is.na(yi.above),"above", # new column indicating the type of biomass used in the analysis
                                        if_else(!is.na(yi.total),"total", "below")))) %>%
  drop_na(SCN) # remove missing data
df%>%group_by(biomass.type)%>%summarise(n())
moderators <- c("MAT","MAP","SCN","Nitrogen","Duration","Tdelta","Ecosystems","Myc","Warming.method","Vegetation.type","biome.type")
newnames <- c("MAT","MAP","soil CN","Nitrogen","Duration","Tdelta","Ecosystems","Mycorrhizal type","Warming method","Vegetation type","Biome type")
names(newnames) <- moderators
df<- dplyr::select(df,yi, vi, Study, moderators) %>% drop_na()
####---------------------------------------------- META-FOREST ----------------------------------------------####
####-------------- OPTIMIZATION --------------###
set.seed(36326)
# Check how many iterations metaforest needs to converge
check_conv <- MetaForest(as.formula(paste0("yi~", paste(moderators, collapse = "+"))),
                         data = df,
                         study = "Study", # dependent data
                         whichweights = "random",
                         num.trees = 100000)
# Plot convergence trajectory
plot(check_conv)
# Perform recursive preselection
set.seed(45)
preselected <- preselect(check_conv, replications = 100, algorithm = "recursive")
pre <- plot(preselected, label_elements = newnames)
pre

# Tune the metaforest analysis

# Set up 10-fold CV
set.seed(728)
cv_folds <- trainControl(method = "cv", 10)
repeatedcv <- trainControl(method="repeatedcv", number=10, repeats=3)
grouped_cv <- trainControl(method = "cv", index = groupKFold(df$Study, k = 10)) # This is the one we'll use for a cluster analysis
# This function will use 75% of the data as "training" data, with the remaining 25% of the data used as "validation" data.

# Set up a tuning grid for the three tuning parameters of MetaForest
# This will make sure we try all possible combinations of hyperparameters to tune the model so we end up with the optimal model
tuning_grid <- expand.grid(whichweights = c("random", "fixed", "unif"),
                           mtry = 2:6,
                           min.node.size = 2:6)

# Select only moderators and vi
# Here we remove part of the moderators based on the preselection analysis we ran before, using a cutoff.
X <- dplyr::select(df, Study,vi, preselect_vars(preselected, cutoff = .5))

# Finally train the model
set.seed(36326)
mf_cv <- train(y = df$yi,
               x = X,
               study = "Study", # dependent data
               method = ModelInfo_mf(),
               trControl = grouped_cv,
               tuneGrid = tuning_grid,
               keep.inbag = TRUE,
               num.trees = 25000)
saveRDS(mf_cv, "mf_cv.RData")
# Check result
# Examine optimal tuning parameters
mf_cv$results[which.min(mf_cv$results$RMSE), ]
# Cross-validated R2 of the final model:
mf_cv$results[which.min(mf_cv$results$RMSE), ]$Rsquared 
# Extract final model
forest <- mf_cv$finalModel
# Plot convergence
plot(forest)

#forest$forest$r.squared
# Plot variable importance
imp.abs <- VarImpPlot(forest,label_elements = newnames)
imp.abs
# Plot partial dependence
arr <- as.list(newnames)
names(arr) <- moderators
mylabel <- function(val) { return(lapply(val, function(x) arr[x])) }

pi <- PartialDependence(forest, vars = names(forest$forest$variable.importance)[order(forest$forest$variable.importance, decreasing = TRUE)], 
                        rawdata = T, pi = 0.95, output = "list",  plot_int = TRUE, bw=TRUE)
p <- lapply(pi, function(x){
  x + facet_wrap(~Variable,labeller=mylabel,ncol = 3) +
    ylab("log response ratio")
})
metaforest:::merge_plots(p)


##################### PREDICTION ##############################
library(raster)
make_pct <- function(x) (exp(x) - 1) * 100
antiperc <- function(x) log(x/100 + 1)
library(parallel)
library(doParallel)
registerDoParallel(31) # Run in parallel -> 31 cores

s<- stack("myStack.grd")
names(s) <- c("biomass","Nitrogen","Ecosystems","biome.type","Vegetation.type", "Myc", "SCN", "MAT", "MAP")
s.df <- as.data.frame(s,xy=TRUE) %>% 
  #rename("Nitrogen"=Nitrogen_category,"Ecosystems"=Ecosystems_category,
   #      "biome.type"=biome.type_category,"Vegetation.type"=Vegetation.type_category) %>%
  mutate(Myc=recode_factor(Myc,"AM", "ECM", "NM"),
         Warming.method=factor("infrared heater", levels=c("Cable","Greenhouse","infrared heater", "OTC", "passive warming")),
         Duration = 5,
         Tdelta = 2)

r.dfNA <- s.df[complete.cases(s.df),]

mod <- readRDS("mf_cv.RData")
forest.abs <- mod$finalModel
number_of_chunks = 200
library(parallel)
library(doParallel)
registerDoParallel(31) # Run in parallel -> 31 cores
preds <- lapply(seq(1, NROW(r.dfNA), ceiling(NROW(r.dfNA)/number_of_chunks)),
                function(i) {
                  df_tmp <- r.dfNA[i:min(i + ceiling(NROW(r.dfNA)/number_of_chunks) - 1, NROW(r.dfNA)),]
                  predict(forest.abs, 
                          type="se",
                          data = df_tmp)
                })
##########  Error in predict.ranger.forest(forest, data, predict.all, num.trees, type,  : 
#Error: No saved inbag counts in ranger object. Please set keep.inbag=TRUE when calling ranger. 

#RF <- data.frame(predictions=unlist(lapply(preds, `[[`, "predictions")), se=unlist(lapply(preds, `[[`, "se")) )
#pred.RF <- cbind(r.dfNA[c("x","y")], RF)
#abs.preds <- left_join(s.df,pred.RF)
#absES <- rasterFromXYZ(abs.preds[,c("x", "y", "predictions")],crs="+proj=longlat +datum=WGS84")
#plot(absES)
#absSE <- rasterFromXYZ(abs.preds[,c("x", "y", "se")],crs="+proj=longlat +datum=WGS84")
#plot(absSE)

# It works if we don't predict SE's
preds <- lapply(seq(1, NROW(r.dfNA), ceiling(NROW(r.dfNA)/number_of_chunks)),
                function(i) {
                  df_tmp <- r.dfNA[i:min(i + ceiling(NROW(r.dfNA)/number_of_chunks) - 1, NROW(r.dfNA)),]
                  predict(forest.abs, 
                          #type="se", 
                          data = df_tmp)})


RF <- data.frame(predictions=unlist(lapply(preds, `[[`, "predictions")))
pred.RF <- cbind(r.dfNA[c("x","y")], RF)
abs.preds <- left_join(s.df,pred.RF)
ES <- make_pct(rasterFromXYZ(abs.preds[,c("x", "y", "predictions")],crs="+proj=longlat +datum=WGS84"))
plot(ES)


