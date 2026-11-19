/* Minimal Foundation stubs for the Linux ObjC syntax gate (make objc-syntax).
 * Declarations only, mirroring the Apple API surface used by the GPU sources.
 * NOT an implementation; never compiled into any real build. */
#ifndef H3I_STUB_FOUNDATION_H
#define H3I_STUB_FOUNDATION_H

#include <stddef.h>
#include <stdint.h>

typedef unsigned long NSUInteger;
typedef long NSInteger;
typedef signed char BOOL;
#define YES ((BOOL)1)
#define NO  ((BOOL)0)
#ifndef nil
#define nil ((id)0)
#endif
#ifndef Nil
#define Nil ((Class)0)
#endif

#define MIN(a, b) ((a) < (b) ? (a) : (b))

@class NSString, NSError;

@protocol NSFastEnumeration
- (NSUInteger)countByEnumeratingWithState:(void *)state
                                  objects:(id *)buffer
                                    count:(NSUInteger)len;
@end

@protocol NSObject
- (BOOL)isEqual:(id)obj;
- (NSUInteger)hash;
- (Class)superclass;
- (Class)class;
- (instancetype)self;
- (instancetype)retain;
- (void)release;
- (instancetype)autorelease;
@end

@interface NSObject <NSObject>
+ (instancetype)new;
+ (instancetype)alloc;
- (instancetype)init;
@end

@interface NSString : NSObject
+ (instancetype)stringWithUTF8String:(const char *)s;
+ (instancetype)stringWithFormat:(NSString *)fmt, ...;
+ (instancetype)stringWithCharacters:(const unsigned short *)chars
                              length:(NSUInteger)len;
+ (instancetype)stringWithContentsOfFile:(NSString *)path
                                encoding:(NSUInteger)enc
                                   error:(NSError **)err;
- (const char *)UTF8String;
- (NSUInteger)length;
@end

@interface NSArray<__covariant ObjectType> : NSObject <NSFastEnumeration>
+ (instancetype)arrayWithObjects:(const id[])objs count:(NSUInteger)cnt;
- (NSUInteger)count;
- (ObjectType)objectAtIndexedSubscript:(NSUInteger)idx;
@end

@interface NSMutableArray<ObjectType> : NSArray<ObjectType>
+ (instancetype)array;
- (void)addObject:(ObjectType)obj;
- (void)removeAllObjects;
@end

@interface NSDictionary<__covariant KeyType, __covariant ObjectType> : NSObject <NSFastEnumeration>
+ (instancetype)dictionaryWithObjects:(const id[])objs
                              forKeys:(const id[])keys
                                count:(NSUInteger)cnt;
- (ObjectType)objectForKeyedSubscript:(KeyType)key;
- (NSUInteger)count;
@end

@interface NSMutableDictionary<KeyType, ObjectType> : NSDictionary<KeyType, ObjectType>
- (void)setObject:(ObjectType)obj forKeyedSubscript:(KeyType)key;
@end

@interface NSNumber : NSObject
+ (NSNumber *)numberWithInt:(int)v;
+ (NSNumber *)numberWithUnsignedInt:(unsigned int)v;
+ (NSNumber *)numberWithLong:(long)v;
+ (NSNumber *)numberWithUnsignedLong:(unsigned long)v;
+ (NSNumber *)numberWithLongLong:(long long)v;
+ (NSNumber *)numberWithUnsignedLongLong:(unsigned long long)v;
+ (NSNumber *)numberWithDouble:(double)v;
+ (NSNumber *)numberWithFloat:(float)v;
+ (NSNumber *)numberWithBool:(BOOL)v;
+ (NSNumber *)numberWithInteger:(NSInteger)v;
+ (NSNumber *)numberWithUnsignedInteger:(NSUInteger)v;
- (NSUInteger)unsignedIntegerValue;
- (int)intValue;
@end

@interface NSData : NSObject
+ (instancetype)dataWithBytes:(const void *)bytes length:(NSUInteger)len;
@end

@interface NSError : NSObject
@property (readonly, copy) NSString *localizedDescription;
@end

@interface NSProcessInfo : NSObject
+ (NSProcessInfo *)processInfo;
@property (readonly) uint64_t physicalMemory;
@end

enum { NSUTF8StringEncoding = 4 };

void *CFBridgingRetain(id obj);
id CFBridgingRelease(void *p);

#endif /* H3I_STUB_FOUNDATION_H */
