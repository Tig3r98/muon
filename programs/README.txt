These programs expect the xxusb libraries to be installed to:
/usr/lib/libxx_usb.so
/usr/include/lubxxusb.h
and since they are based on libusb will require root privilegers
you can bypass that by creating
/etc/udev/rules.d/cc_usb.rules
which must contain this line:
SUBSYSTEM="usb", ATTR{idVendor}=="16dc", ATTR{idProduct}=="0001", MODE="0666"
This bypasses all requirements (0666) for using Wiener (16dc) CC_USB (0001) devices.
Compilation is made by executing:
gcc program.c -o program -lxx_usb -lusb

To get the hadle which you'll need in xxusb functions, you can use this:

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

The version of the library used was 3.22, but it'd be better to update it.
