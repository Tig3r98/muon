//****************************************************************************
/*    ADC
gcc ADC.c -Wall -o ADC -lxx_usb -lusb

Descrizione:
Il programma legge e stampa (a terminale e su file) gli 8 registri dell'
ADC posto nella station numerata con ADC_SLOT.
La misura dura in totale TIME, mentre il singolo dato ha a disposizione
una finestra temporale di WAIT.

*/
//****************************************************************************

// N è il numero della stazione nel crate (0-24)
// A è il subadress (usato per moduli a più sezioni)
// F è il numero della funzione

#define ADC_SLOT 8
//#define TIME 100*WAIT*300		//tempo totale di misura in microsecondi
#define WAIT 10000				//tempo per ogni singola misura
#define CHANNELS 12

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <errno.h>
#include <ctype.h>

#include <libxxusb.h>

char *cszPrgName;
usb_dev_handle* handle;

//find camac
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

int main(int argc, char **argv){
  //variabili necessaria
  char *fname = DEVICE_NAME;
  long int i , j = 0;
  int error = 0;
  long int k, w = 0;
  int q, x = 0;

  // apertura del file di output dei dati
  FILE *dati = fopen("./dati/datiADC.txt","w");
  if (dati == NULL){
    perror("Errore apertura file: manca la cartella?");
    exit(1);
  }
  
  //apre un crate usb
  handle = find_devices();
  if(handle == 0){
    fprintf(stderr, "%s: %s: %s\n", cszPrgName, fname, strerror(error));
    exit(1);
  } else{
    printf("%s: open done.\n", cszPrgName);
  }
  
  //disable LAM
  //CAMAC_read(handle,ADC_SLOT,0,24,&w,&q,&x);

  //resetta tutti i canali dell'ADC
  for(i = 0; i < 12; i++){
    CAMAC_read(handle,ADC_SLOT,i,2,&w,&q,&x);
  }

  //for (i = 0; i < (2*TIME); i++){
  i = 0;
  while (i < 3000){
  
    CAMAC_read(handle,ADC_SLOT,0,0,&w,&q,&x);
    if (/*q == 1 && x == 1*/1){
      usleep(WAIT);
      
      int values[12];
      int error = 0;
      for (j = 0; j < CHANNELS; j++) {             //per ciascun canale stampa
        CAMAC_read(handle,ADC_SLOT,j,2,&k,&q,&x);
        values[j] = k;
        //select interesting channels
        if(j == 8 || j == 9){
          if(x!=1){
            //printf("CR \n"); //COMMAND REJECTED
            error = 1;
          }
          if(q==0){
            //printf("NR \n"); //NO RESPONSE
            error = 1;
          }
          if(k == 0){
            //printf("0 \n"); //ZERO
            error = 1;
          }
        }
      }
      if(error)
        continue;
      printf("%d ", i); //progress
      for (j = 8; j < 10; j++) {
        printf("%ld ", values[j]);
        fprintf(dati, "%ld ", values[j]);
      }
      printf("\n");
      fprintf(dati, "\n");
      i++;
      CAMAC_read(handle,ADC_SLOT,0,10,&k,&q,&x);
      //printf("\n");
      
          
      //printf("TIME: %f. ",(float)i*WAIT/1000000);
      //printf("ADC counts: ");
/*      long int vals[CHANNELS]; 
      int allZero = 1;
      for (j = 0; j < CHANNELS; j++) {             //per ciascun canale stampa
        k = cc32_read_word(handle,ADC_SLOT,j,2);
        vals[j] = k;
        if( k != 0 ) allZero = 0;
      }
	
      if (i*WAIT >= TIME) break;  //se si è superato TIME si esce
      if( allZero == 0 )
      { 
        for (j = 0; j < CHANNELS; j++) {             //per ciascun canale stampa 
          printf("%ld ",vals[j]);
          fprintf(dati, "%ld ", vals[j]);   
          }
         printf("\n");
          fprintf(dati, "\n");
 
	}*/
        CAMAC_read(handle,ADC_SLOT,0,10,&k,&q,&x);
 	}  
}

  //stampa dei risultati
  printf("\n");


  //chiude il canale di comunicazione e il file
  xxusb_device_close(handle);
  printf("%s: close done.\n", cszPrgName);
  fclose(dati);

  return 0;
}
