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

#define DEVICE_NAME "/dev/cc32_1"
#define ADC_SLOT 7
#define TIME 100*WAIT*300           //tempo totale di misura in microsecondi
#define WAIT 10000                //tempo per ogni singola misura
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
  
  //enable LAM
  CAMAC_read(handle,ADC_SLOT,0,26,&w,&q,&x);
  printf("enable LAM q %d", q);

  //resetta tutti i canali dell'ADC
  for(i = 0; i < 12; i++){
    CAMAC_read(handle,ADC_SLOT,i,2,&w,&q,&x);
  }

  i = 0;
  while (i < 200){
  
    //channel 0
    //test lam
    int res = CAMAC_read(handle,ADC_SLOT,0,8,&k,&q,&x);
    //printf("res %d q %d, x %d\n", res, q, x);
    if(q == 1){
      printf("ok\n");
      //read register, clear lam
      CAMAC_read(handle,ADC_SLOT,0,2,&k,&q,&x);
      if(k == 0){
        printf("0 \n"); //ZERO
        continue;
      }
      if(q==0){
        printf("NR \n"); //NO RESPONSE
        continue;
      }
      if(x!=1){
        printf("CR \n"); //COMMAND REJECTED
        continue;
      }
      printf("%ld ", k);
      printf("\n");
      fprintf(dati, "%ld ", k);
      fprintf(dati, "\n");
      //clear
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
