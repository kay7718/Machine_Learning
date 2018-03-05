## Detailed Data Cleaning/Visualization
library(data.table)
library(dplyr)
library(tidyr)
library(lubridate)
library(ggplot2)

### ggplot2 Theme Trick
my_theme <- theme_bw() +
  theme(axis.title=element_text(size=24),
        plot.title=element_text(size=36),
        axis.text =element_text(size=16))

my_theme_dark <- theme_dark() +
  theme(axis.title=element_text(size=24),
        plot.title=element_text(size=36),
        axis.text =element_text(size=16))

## First Glance
set.seed(1)
df <- fread("~/data/kaggle/santander/train_ver2.csv", nrows=-1)
unique.id <- unique(df$ncodpers)
limit.people <- 3.5e5
unique.id <- unique.id[sample(length(unique.id),limit.people)]
df <- df[df$ncodpers %in% unique.id,]
str(df)

df$fecha_dato <- as.POSIXct(strptime(df$fecha_dato,format="%Y-%m-%d"))
df$fecha_alta <- as.POSIXct(strptime(df$fecha_alta,format="%Y-%m-%d"))
unique(df$fecha_dato)

df$month <- month(df$fecha_dato)

sapply(df,function(x)any(is.na(x)))

## Data Cleaning
ggplot(data=df,aes(x=age)) + 
  geom_bar(alpha=0.75,fill="tomato",color="black") +
  ggtitle("Age Distribution") + 
  my_theme

df$age[(df$age < 18)]  <- mean(df$age[(df$age >= 18) & (df$age <=30)],na.rm=TRUE)
df$age[(df$age > 100)] <- mean(df$age[(df$age >= 30) & (df$age <=100)],na.rm=TRUE)
df$age[is.na(df$age)]  <- median(df$age,na.rm=TRUE)
df$age                 <- round(df$age)

ggplot(data=df,aes(x=age)) + 
  geom_bar(alpha=0.75,fill="tomato",color="black") +
  xlim(c(18,100)) + 
  ggtitle("Age Distribution") + 
  my_theme

sum(is.na(df$ind_nuevo))

months.active <- df[is.na(df$ind_nuevo),] %>%
  group_by(ncodpers) %>%
  summarise(months.active=n())  %>%
  select(months.active)
max(months.active)

df$ind_nuevo[is.na(df$ind_nuevo)] <- 1 

sum(is.na(df$antiguedad))

summary(df[is.na(df$antiguedad),]%>%select(ind_nuevo))

df$antiguedad[is.na(df$antiguedad)] <- min(df$antiguedad,na.rm=TRUE)
df$antiguedad[df$antiguedad<0]      <- 0

df$fecha_alta[is.na(df$fecha_alta)] <- median(df$fecha_alta,na.rm=TRUE)

table(df$indrel)

df$indrel[is.na(df$indrel)] <- 1

df <- df %>% select(-tipodom,-cod_prov)

sapply(df,function(x)any(is.na(x)))

sum(is.na(df$ind_actividad_cliente))

df$ind_actividad_cliente[is.na(df$ind_actividad_cliente)] <- median(df$ind_actividad_cliente,na.rm=TRUE)

unique(df$nomprov)

df$nomprov[df$nomprov==""] <- "UNKNOWN"

sum(is.na(df$renta))

df %>%
  filter(!is.na(renta)) %>%
  group_by(nomprov) %>%
  summarise(med.income = median(renta)) %>%
  arrange(med.income) %>%
  mutate(city=factor(nomprov,levels=nomprov)) %>% # the factor() call prevents reordering the names
  ggplot(aes(x=city,y=med.income)) + 
  geom_point(color="#c60b1e") + 
  guides(color=FALSE) + 
  xlab("City") +
  ylab("Median Income") +  
  my_theme + 
  theme(axis.text.x=element_blank(), axis.ticks = element_blank()) + 
  geom_text(aes(x=city,y=med.income,label=city),angle=90,hjust=-.25) +
  theme(plot.background=element_rect(fill="#c60b1e"),
        panel.background=element_rect(fill="#ffc400"),
        panel.grid =element_blank(),
        axis.title =element_text(color="#ffc400"),
        axis.text  =element_text(color="#ffc400"),
        plot.title =element_text(color="#ffc400",size=32)) +
  ylim(c(50000,200000)) +
  ggtitle("Income Distribution by City")

new.incomes <-df %>%
  select(nomprov) %>%
  merge(df %>%
          group_by(nomprov) %>%
          summarise(med.income=median(renta,na.rm=TRUE)),by="nomprov") %>%
  select(nomprov,med.income) %>%
  arrange(nomprov)
df <- arrange(df,nomprov)
df$renta[is.na(df$renta)] <- new.incomes$med.income[is.na(df$renta)]
rm(new.incomes)

df$renta[is.na(df$renta)] <- median(df$renta,na.rm=TRUE)
df <- arrange(df,fecha_dato)

sum(is.na(df$ind_nomina_ult1))

df[is.na(df)] <- 0

str(df)

char.cols <- names(df)[sapply(df,is.character)]
for (name in char.cols){
  print(sprintf("Unique values for %s:", name))
  print(unique(df[[name]]))
  cat('\n')
}

df$indfall[df$indfall==""]                 <- "N"
df$tiprel_1mes[df$tiprel_1mes==""]         <- "A"
df$indrel_1mes[df$indrel_1mes==""]         <- "1"
df$indrel_1mes[df$indrel_1mes=="P"]        <- "5" # change to just numbers because it currently contains letters and numbers
df$indrel_1mes                             <- as.factor(as.integer(df$indrel_1mes))
df$pais_residencia[df$pais_residencia==""] <- "UNKNOWN"
df$sexo[df$sexo==""]                       <- "UNKNOWN"
df$ult_fec_cli_1t[df$ult_fec_cli_1t==""]   <- "UNKNOWN"
df$ind_empleado[df$ind_empleado==""]       <- "UNKNOWN"
df$indext[df$indext==""]                   <- "UNKNOWN"
df$indresi[df$indresi==""]                 <- "UNKNOWN"
df$conyuemp[df$conyuemp==""]               <- "UNKNOWN"
df$segmento[df$segmento==""]               <- "UNKNOWN"

features          <- grepl("ind_+.*ult.*",names(df))
df[,features]     <- lapply(df[,features],function(x)as.integer(round(x)))
df$total.services <- rowSums(df[,features],na.rm=TRUE)

df               <- df %>% arrange(fecha_dato)
df$month.id      <- as.numeric(factor((df$fecha_dato)))
df$month.next.id <- df$month.id + 1

status.change <- function(x){
  if ( length(x) == 1 ) { # if only one entry exists, I'll assume they are a new customer and therefore are adding services
    label = ifelse(x==1,"Added","Maintained")
  } else {
    diffs <- diff(x) # difference month-by-month
    diffs <- c(0,diffs) # first occurrence will be considered Maintained, which is a little lazy. A better way would be to check if the earliest date was the same as the earliest we have in the dataset and consider those separately. Entries with earliest dates later than that have joined and should be labeled as "Added"
    label <- rep("Maintained", length(x))
    label <- ifelse(diffs==1,"Added",
                    ifelse(diffs==-1,"Dropped",
                           "Maintained"))
  }
  label
}

df[,features] <- lapply(df[,features], function(x) return(ave(x,df$ncodpers, FUN=status.change)))

interesting <- rowSums(df[,features]!="Maintained")
df          <- df[interesting>0,]
df          <- df %>%
  gather(key=feature,
         value=status,
         ind_ahor_fin_ult1:ind_recibo_ult1)
df          <- filter(df,status!="Maintained")
head(df)

## Data Visualizations

totals.by.feature <- df %>%
  group_by(month,feature) %>%
  summarise(counts=n())

df %>% 
  group_by(month,feature,status) %>%
  summarise(counts=n())%>%
  ungroup() %>%
  inner_join(totals.by.feature,by=c("month","feature")) %>%
  
  mutate(counts=counts.x/counts.y) %>%
  ggplot(aes(y=counts,x=factor(month.abb[month],levels=month.abb[seq(12,1,-1)]))) +
  geom_bar(aes(fill=status), stat="identity") +
  facet_wrap(facets=~feature,ncol = 6) +
  coord_flip() +
  my_theme_dark + 
  ylab("Count") +
  xlab("") + 
  ylim(limits=c(0,1)) +
  ggtitle("Relative Service \nChanges by Month") +
  theme(axis.text   = element_text(size=10),
        legend.text = element_text(size=14),
        legend.title= element_blank()      ,
        strip.text  = element_text(face="bold")) +
  scale_fill_manual(values=c("cyan","magenta"))

month.counts              <- table(unique(df$month.id)%%12)
cur.names                 <- names(month.counts)
cur.names[cur.names=="0"] <- "12"
names(month.counts) <- cur.names
month.counts              <- data.frame(month.counts) %>%
  rename(month=Var1,month.count=Freq) %>% mutate(month=as.numeric(month))

df %>% 
  group_by(month,feature,status) %>%
  summarise(counts=n())%>%
  ungroup() %>%
  inner_join(month.counts,by="month") %>%
  
  mutate(counts=counts/month.count) %>%
  ggplot(aes(y=counts,x=factor(month.abb[month],levels=month.abb[seq(12,1,-1)]))) +
  geom_bar(aes(fill=status), stat="identity") +
  facet_wrap(facets=~feature,ncol = 6) +
  coord_flip() +
  my_theme_dark + 
  ylab("Count") +
  xlab("") + 
  ggtitle("Average Service \nChanges by Month") +
  theme(axis.text    = element_text(size=10),
        legend.text  = element_text(size=14),
        legend.title = element_blank()      ,
        strip.text   = element_text(face="bold")) +
  scale_fill_manual(values=c("cyan","magenta"))

df %>%
  filter(sexo!="UNKNOWN") %>%
  ggplot(aes(x=sexo)) +
  geom_bar(aes(fill=status)) +
  facet_wrap(facets=~feature,ncol = 6) +
  my_theme_dark + 
  ylab("Count") +
  xlab("") +
  ggtitle("Service Changes by Gender") +
  theme(axis.text    = element_text(size=10),
        legend.text  = element_text(size=14),
        legend.title = element_blank()      ,
        strip.text   = element_text(face="bold")) +
  scale_fill_manual(values=c("cyan","magenta"))

tot.H  <- sum(df$sexo=="H")
tot.V  <- sum(df$sexo=="V")
tmp.df <- df %>%
  group_by(sexo,status) %>%
  summarise(counts=n())
tmp.df$counts[tmp.df$sexo=="H"] = tmp.df$counts[tmp.df$sexo=="H"] / tot.H
tmp.df$counts[tmp.df$sexo=="V"] = tmp.df$counts[tmp.df$sexo=="V"] / tot.V
tmp.df %>%
  filter(sexo!="UNKNOWN") %>%
  ggplot(aes(x=factor(feature),y=counts)) +
  geom_bar(aes(fill=status,sexo),stat='identity') +
  coord_flip() +
  my_theme_dark + 
  ylab("Ratio") +
  xlab("") +
  ggtitle("Normalized Service \n Changes by Gender") +
  theme(axis.text    = element_text(size=20),
        legend.text  = element_text(size=14),
        legend.title = element_blank()      ,
        strip.text   = element_text(face="bold")) +
  scale_fill_manual(values=c("cyan","magenta"))

rm(tmp.df)

tot.new     <- sum(df$ind_nuevo==1)
tot.not.new <- sum(df$ind_nuevo!=1)
tmp.df      <- df %>%
  group_by(ind_nuevo,status) %>%
  summarise(counts=n())
tmp.df$counts[tmp.df$ind_nuevo==1] = tmp.df$counts[tmp.df$ind_nuevo==1] / tot.new
tmp.df$counts[tmp.df$ind_nuevo!=1] = tmp.df$counts[tmp.df$ind_nuevo!=1] / tot.not.new
tmp.df %>%
  ggplot(aes(x=factor(feature),y=counts)) +
  geom_bar(aes(fill=status,factor(ind_nuevo)),stat='identity') +
  coord_flip() +
  my_theme_dark + 
  ylab("Count") +
  xlab("") +
  ggtitle("Normalized Service \n Changes by New Status") +
  theme(axis.text    = element_text(size=10),
        legend.text  = element_text(size=14),
        legend.title = element_blank()      ,
        strip.text   = element_text(face="bold")) +
  scale_fill_manual(values=c("cyan","magenta"))

rm(tmp.df)

df %>%
  group_by(nomprov,status) %>%
  summarise(y=mean(total.services)) %>%
  ggplot(aes(x=factor(nomprov,levels=sort(unique(nomprov),decreasing=TRUE)),y=y)) +
  geom_bar(stat="identity",aes(fill=status)) +
  geom_text(aes(label=nomprov),
            y=0.2,
            hjust=0,
            angle=0,
            size=3,
            color="#222222") +
  coord_flip() +
  my_theme_dark +
  xlab("City") +
  ylab("Total # Changes") + 
  ggtitle("Service Changes\n by City") +
  theme(axis.text    = element_blank(),
        legend.text  = element_text(size=14),
        legend.title = element_text(size=18)) +
  scale_fill_manual(values=c("cyan","magenta"))

df %>%
  group_by(antiguedad,status) %>%
  summarise(counts=n()) %>%
  ggplot(aes(x=factor(antiguedad),y=log(counts))) +
  geom_point(alpha=0.6,aes(color=status)) +
  my_theme_dark +
  xlab("Seniority (Months)") +
  ylab("Total # Changes") + 
  ggtitle("Service Changes \n by Seniority") +
  theme(axis.text    = element_blank(),
        legend.text  = element_text(size=14),
        legend.title = element_text(size=18)) +
  scale_color_manual(values=c("cyan","magenta"))

df %>%
  ggplot(aes(x=age,y=log(renta))) +
  geom_point(alpha=0.5,aes(color=status)) +
  my_theme_dark +
  xlab("Age") +
  ylab("Income (log scale)") + 
  ggtitle("Income vs. Age") +
  theme(
    legend.text  = element_text(size=14),
    legend.title = element_text(size=18)) +
  scale_color_manual(values=c("cyan","magenta"))

df %>%
  group_by(ncodpers) %>%
  summarise(age=max(age),seniority=max(antiguedad)) %>%
  select(age,seniority) %>%
  ggplot(aes(x=age,y=seniority)) +
  geom_point(alpha=0.4) +
  ggtitle("Seniority vs. Age") + 
  my_theme

df %>%
  group_by(nomprov,status) %>%
  summarise(y=mean(total.services)) %>%
  ggplot(aes(x=factor(nomprov,levels=sort(unique(nomprov),decreasing=TRUE)),y=y)) +
  geom_bar(stat="identity",aes(fill=status)) +
  geom_text(aes(label=nomprov),
            y=0.2,
            hjust=0,
            angle=0,
            size=3,
            color="#222222") +
  coord_flip() +
  my_theme_dark +
  xlab("City") +
  ylab("Total # Changes") + 
  ggtitle("Service Changes\n by City") +
  theme(axis.text    = element_blank(),
        legend.text  = element_text(size=14),
        legend.title = element_text(size=18)) +
  scale_fill_manual(values=c("cyan","magenta"))

