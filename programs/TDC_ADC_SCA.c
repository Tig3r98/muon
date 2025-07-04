//**************************************************************************
/*    TDC ADC SCA
gcc TDC_ADC_SCA.c -Wall -o TDC_ADC_SCA -lxx_usb -lusb

Descrizione:
Effettua N_MEAS misure di MEAS_TIME secondi in maniera analoga a TDC_ADC.
Gli eventi durante questi MEAS_TIME scondi vengono salvati su un file.
Per separare le 3 misure, viene inserita una riga:
EOM SCA1 SCA2 SCA3 ...
"EOM" sta per end of measure ed è seguita dai valori dei canali dello scaler,
che vengono resettati ogni misura.

*/
//***************************************************************************

// N è il numero della stazione nel crate (0-24)
// A è il subadress (usato per moduli a più sezioni)
// F è il numero della funzione
// Q è lo status della risposta della station (0 o 1)
// X è lo status dell'accettazione del comando (0 o 1)

#define SCA_SLOT 21  // slot dello scaler
#define TDC_SLOT 12  // slot del TDC	
#define ADC_SLOT 8   // slot dell'ADC
#define WAIT 108       // espresso in microsecondi, il tempo di digitalizzazione dell'ADC 
#define N_MEAS 3      // numero di misure 
#define MEAS_TIME 300  // tempo di acquisizione per ogni misura, secondi
#define ADC_CHAN_1 8  
#define ADC_CHAN_2 9

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <errno.h>
#include <ctype.h>
#include <time.h>

#include <libxxusb.h>

//find camac
#define XXUSB_CCUSB_PRODUCT_ID 0x0001
usb_dev_handle* find_devices(){
    //maximum number of crates: 128
    xxusb_device_type devices[128];
    
    short count = xxusb_devices_find(devices);
    if (count < 0)
      printf("Couldn't get list of XXUSB (crate) devices.");
    
    for (int i=0; i<count; ++i) {
        auto* dev = devices[i].usbdev;
        //open device
        usb_dev_handle* handle = xxusb_device_open(dev);
        //verify correct opening
        if(!handle)
            return 0;
        return handle;
    }
    return 0;
}

usb_dev_handle* handle;
char *cszPrgName;

int main(int argc, char **argv){
  // Variabili necessarie
  char *fname = "dev/ccusb";
  int q, x = 0;
  long int j ;
  int error;
  int ich;
  long int k = 0;
  
  //disable buffering, if the program crashes data has already been written
  setbuf(stdout, NULL);

  // apertura del file di output dei dati
  FILE *dati = fopen("./dati/datitime.txt","w");
  if (dati == NULL){
    perror("Errore apertura file: manca la cartella?");
    exit(1);
  }
  
  //apertura crate usb
  handle = find_devices();
  if(handle == 0){
    fprintf(stderr, "%s: %s: %s\n", cszPrgName, fname, strerror(error));
    exit(1);
  } else{
    printf("%s: open done.\n", cszPrgName);
  }
  
  // Init crate
  CAMAC_Z(handle);
  // TODO needed?
  // Clear TDC
  //handle, N, A, F, data, q, x
  CAMAC_read(handle,TDC_SLOT,0,9,&k,&q,&x); //pulizia registri
  CAMAC_read(handle,TDC_SLOT,0,10,&k,&q,&x); //pulizia LAM
  // Clear ADC
  int i;
  for(i = 0; i < 12; i++){
    CAMAC_read(handle,ADC_SLOT,i,2,&k,&q,&x);
  }
  
  //flags for when a signal has been received signal
  int good_tdc, good_adc1, good_adc2 = 0;
  int tdc_data, adc1_data, adc2_data = 0;
  
  
    
  for(int n_meas_index=0; n_meas_index<N_MEAS; n_meas_index++){
    int count = 0;
  
    //dumb way to wait for the second start
    //printf("Waiting for second start...\n");
    unsigned long target_start_time = time(NULL) + 1;
    while(time(NULL) < target_start_time){}
    
    unsigned long start_time = time(NULL);
    printf("MEASURE N%d\n", n_meas_index+1);
    // Clear SCA
    for (i = 0; i < 12; i++) CAMAC_read(handle,SCA_SLOT,i,2,&k,&q,&x);
    while((unsigned long)time(NULL) - start_time < MEAS_TIME){
      //read and clear channel 0 register
      //CAMAC_read(handle,TDC_SLOT,0,2,&k,&q,&x);
      //CAMAC_read(handle,ADC_SLOT,0,0,&k,&q,&x);
      
      //wait because yes I guess
      //usleep(((unsigned long int)WAIT)*100);
      
      //READ DATA
      if(!good_tdc){
        CAMAC_read(handle,TDC_SLOT,0,2,&k,&q,&x);
        tdc_data = k;
        if(q && x) good_tdc = 1;
        //typically the ADC is slower, we could just wait a bit here.
        //The LeCroy 2259 has a 108usec conversion time.
        usleep(108);
      }
      if(!good_adc1){
        CAMAC_read(handle,ADC_SLOT,ADC_CHAN_1,2,&k,&q,&x);
        adc1_data = k;
        if(q && x && adc1_data) good_adc1 = 1;        
      }
      if(!good_adc2){
        CAMAC_read(handle,ADC_SLOT,ADC_CHAN_2,2,&k,&q,&x);
        adc2_data = k;
        if(q && x && adc2_data) good_adc2 = 1;
      }
      
      //if not all signals have been compiled, repeat asks
      if(!(good_tdc && good_adc1 && good_adc2)){
        continue;
      } else {
        //otherwise reset flags
        good_tdc = 0;
        good_adc1 = 0;
        good_adc2 = 0;
      }
    
      //if either ADC is 0, skip datapoint
      //if(!(adc1_data | adc2_data)) continue; 
      
      count++;
      printf("T%lu\t", (unsigned long)time(NULL) - start_time); //time elapsed
      printf("C%d\t", count); //tdc_adc counts
      //print TDC
      printf("TDC%ld\t\t",tdc_data);
      fprintf(dati, "%ld ",tdc_data);
      //print ADC
      printf("ADC%ld\t",adc1_data);
      fprintf(dati, "%ld ",adc1_data);
      printf("ADC%ld\n",adc2_data);
      fprintf(dati, "%ld\n",adc2_data);
    
      //flush file
      fflush(dati);
      
      //reset data
      tdc_data = 0;
      adc1_data = 0;
      adc2_data = 0;
    
      //clear TDC and ADC by using function 2 on the last channels
      CAMAC_read(handle,TDC_SLOT,7,2,&k,&q,&x); //not that nice...
      CAMAC_read(handle,ADC_SLOT,11,2,&k,&q,&x);
    }
    printf("EOM");
    fprintf(dati, "EOM"); //EOM = end of measure
    //print scaler data
    for (j = 0; j < 12; j++) {                 // per i 12 registri dello scaler
      CAMAC_read(handle,SCA_SLOT,j,0,&k,&q,&x); // lettura registri e stampa
      if(k != 0) printf("\tSCA%d:%ld",j,k); //only print when different than 0 to not pollute screen
      fprintf(dati, " %ld",k); //only print when different than 0 to not pollute screen
    }
    printf("\n");
    fprintf(dati, "\n");
  }
  

  //chiude il canale di comunicazione e il file
  xxusb_device_close(handle);
  printf("%s: close done.\n", cszPrgName);
  fclose(dati);

  return 0;
}
