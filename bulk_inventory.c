// Read-only macOS directory metadata enumeration. No write/delete operations.
// Attribute packing follows Apple's getattrlistbulk(2) and getattrlist(2).
#include <sys/attr.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>

#pragma pack(push,4)
struct entry {
    uint32_t length;
    attribute_set_t returned;
    attrreference_t name;
    uint32_t object_type;
    struct timespec modified;
    uint64_t file_id;
    off_t data_length;
};
#pragma pack(pop)

int main(int argc,char **argv) {
    if(argc!=2) return 2;
    int fd=open(argv[1],O_RDONLY|O_DIRECTORY|O_NOFOLLOW);
    if(fd<0) {perror("open");return 1;}
    struct attrlist attrs={0};
    attrs.bitmapcount=ATTR_BIT_MAP_COUNT;
    attrs.commonattr=ATTR_CMN_RETURNED_ATTRS|ATTR_CMN_NAME|ATTR_CMN_OBJTYPE|ATTR_CMN_MODTIME|ATTR_CMN_FILEID;
    attrs.fileattr=ATTR_FILE_DATALENGTH;
    char *buffer=malloc(1024*1024);
    if(!buffer) {close(fd);return 1;}
    for(;;) {
        int n=getattrlistbulk(fd,&attrs,buffer,1024*1024,FSOPT_PACK_INVAL_ATTRS);
        if(n<0) {perror("getattrlistbulk");free(buffer);close(fd);return 1;}
        if(n==0) break;
        char *cursor=buffer;
        for(int i=0;i<n;i++) {
            struct entry *e=(struct entry*)cursor;
            if(cursor+sizeof(*e)>buffer+1024*1024 || e->length<sizeof(*e) || cursor+e->length>buffer+1024*1024) return 3;
            char *next=cursor+e->length;
            uint32_t needed=attrs.commonattr & ~ATTR_CMN_RETURNED_ATTRS;
            if(e->object_type==1) { // VREG
                if((e->returned.commonattr&needed)!=needed || !(e->returned.fileattr&ATTR_FILE_DATALENGTH)) return 4;
                char *name=(char*)&e->name+e->name.attr_dataoffset;
                uint32_t length=e->name.attr_length;
                if(length==0 || name<cursor || name+length>next || name[length-1]!='\0') return 5;
                char *q=name;
                while(isdigit((unsigned char)*q)) q++;
                if(q!=name && (!strcmp(q,".jpg") || !strcmp(q,".feather"))) {
                    long long mtime=(long long)e->modified.tv_sec*1000000000LL+e->modified.tv_nsec;
                    printf("%s\t%lld\t%lld\t%llu\n",name,(long long)e->data_length,mtime,(unsigned long long)e->file_id);
                }
            }
            cursor=next;
        }
    }
    free(buffer);close(fd);
    return 0;
}
