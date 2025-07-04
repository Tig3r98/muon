//**************************************************************************
/*    TDC
gcc TDC_lecroy.c -Wall -o TDC -lxx_usb -lusb

Descrizione:
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

#define TDC_SLOT 16  // slot del tdc			
#define WAIT 1       // espresso in usecondi
#define EVENTS 100   // numero massimo di eventi

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
  char *fname = "dev/ccusb";
  int q, x = 0;
  long int j ;
  int error;
  int ich;
  long int k = 0;

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

  // Clear module
  for(j = 0; j < 8; j++){
    CAMAC_read(handle,TDC_SLOT,j,9,&k,&q,&x); //pulizia registri
  }
  CAMAC_read(handle,TDC_SLOT,0,10,&k,&q,&x); //pulizia LAM

  int count = 0;
  while(count < EVENTS){
    usleep(((unsigned long int)WAIT)*100000);
    //read and clear channel 0 register
    
    CAMAC_read(handle,TDC_SLOT,0,2,&k,&q,&x);

    // se il comando è accettato entra nell'if
    if (q == 1 && x == 1) {
      count++;
      printf("C%d ", count);
      for (ich = 0; ich < 8; ich++){  //per ogni canale del TDC stampa i registri
        //la funzione 2 legge e pulisce i registri
        //input: handle, N, A, F
        CAMAC_read(handle,TDC_SLOT,ich,2,&k,&q,&x);
	printf("%ld ",k);
        if (ich == 0){
          fprintf(dati, "%ld\n", k);
        }
        
      }      
      printf("\n");
    }

    //Azzera il test LAM, probabilmente inutile
    //CAMAC_read(handle,TDC_SLOT,0,10,&k,&q,&x);
  }

  //chiude il canale di comunicazione e il file
  xxusb_device_close(handle);
  printf("%s: close done.\n", cszPrgName);
  fclose(dati);

  return 0;
}
