//**************************************************************************
/*    TDC
gcc TDC.c -Wall -o TDC -lxx_usb -lusb

Descrizione:
Il programma legge e stampa (a terminale e su file) gli 8 registri del TDC
posto nella station numerata con TDC_SLOT.
La misura dura un massimo tempo TIME o qualora si raggiunga un numero
di misure riuscite superiore a MAXEV.
Ogni WAIT microsecondi vengono letti e azzerati i registri perciò è
importante che WAIT sia > dei ritardi in gioco.

*/
//***************************************************************************

// N è il numero della stazione nel crate (0-24)
// A è il subadress (usato per moduli a più sezioni)
// F è il numero della funzione
// Q è lo status della risposta della station (0 o 1)
// X è lo status dell'accettazione del comando (0 o 1)

#define DEVICE_NAME "/dev/cc32_1"
#define TDC_SLOT 16			
#define TIME (unsigned long int)1000 // espresso in usecondi
#define WAIT (unsigned long int)1   // espresso in usecondi
#define MAXEV 2000000    // numero massimo di eventi

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <errno.h>
#include <ctype.h>

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
  char *fname = DEVICE_NAME;
  int q, x = 0;
  long k = 0;
  int error = 0;

  // apertura del file di output dei dati
  FILE *dati = fopen("./dati/datitdc.txt","w");
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
  
  //camac init; pulisce i registri
  CAMAC_Z(handle);
  //required: init lam status enable/disable (F25/F24) and lam clear (F10) commands
  CAMAC_read(handle,TDC_SLOT,0,24,&k,&q,&x); //disable lam
  CAMAC_read(handle,TDC_SLOT,0,10,&k,&q,&x); //clear lam
  //CAMAC_read(handle,TCD_SLOT,0,8,&k,&q,&x); //test lam. Q is generated if LAM is true.
  
  while(1){
    //run trough channels
    for(int a=0; a<1; a++){
      //test lam. It is generated on end of conversion
      //CAMAC_read(handle,TDC_SLOT,a,8,&k,&q,&x); //test lam. Q is generated if LAM is true.
      if(1){
        //printf("DEBUG: accepted LAM signal on channel %d\n", a);
        //read register
        CAMAC_read(handle,TDC_SLOT,a,2,&k,&q,&x); //result is stored in k.
        if(q !=0) printf("C%d %ld\n", a, k);
      }
    }
  }

  //chiude il canale di comunicazione e il file
  xxusb_device_close(handle);
  printf("%s: close done.\n", cszPrgName);
  fclose(dati);

  return 0;
}
