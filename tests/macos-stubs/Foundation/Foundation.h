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
typedef unsigned short unichar;
#define YES ((BOOL)1)
#define NO  ((BOOL)0)
#define NSIntegerMax LONG_MAX
#define NSUIntegerMax ULONG_MAX
#include <limits.h>

typedef struct { NSUInteger location, length; } NSRange;
typedef NSInteger NSComparisonResult;
enum {
    NSOrderedAscending = -1,
    NSOrderedSame = 0,
    NSOrderedDescending = 1,
};
#define NSNotFound NSIntegerMax
#ifndef nil
#define nil ((id)0)
#endif
#ifndef Nil
#define Nil ((Class)0)
#endif

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

static inline NSRange NSMakeRange(NSUInteger loc, NSUInteger len) {
    NSRange r = {loc, len};
    return r;
}
static inline NSUInteger NSMaxRange(NSRange r) { return r.location + r.length; }
typedef NSComparisonResult (^NSComparator)(id obj1, id obj2);
enum { NSLiteralSearch = 2, NSCaseInsensitiveSearch = 1, NSBackwardsSearch = 4 };

@class NSString, NSMutableString, NSArray, NSMutableArray, NSDictionary,
       NSMutableDictionary, NSData, NSMutableData, NSError, NSValue, NSNumber;

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
- (id)copy;
- (id)mutableCopy;
- (BOOL)isKindOfClass:(Class)cls;
- (BOOL)isMemberOfClass:(Class)cls;
- (BOOL)respondsToSelector:(SEL)sel;
@end

@interface NSNull : NSObject
+ (NSNull *)null;
@end

@interface NSString : NSObject
+ (instancetype)stringWithUTF8String:(const char *)s;
+ (instancetype)stringWithFormat:(NSString *)fmt, ...;
+ (instancetype)stringWithCharacters:(const unichar *)chars
                              length:(NSUInteger)len;
+ (instancetype)stringWithContentsOfFile:(NSString *)path
                                encoding:(NSUInteger)enc
                                   error:(NSError **)err;
- (const char *)UTF8String;
- (NSUInteger)length;
- (unichar)characterAtIndex:(NSUInteger)idx;
- (void)getCharacters:(unichar *)buffer range:(NSRange)range;
- (NSString *)substringFromIndex:(NSUInteger)idx;
- (NSString *)substringToIndex:(NSUInteger)idx;
- (NSString *)substringWithRange:(NSRange)range;
- (NSString *)stringByAppendingString:(NSString *)s;
- (NSString *)stringByAppendingFormat:(NSString *)fmt, ...;
- (NSComparisonResult)compare:(NSString *)s;
- (NSComparisonResult)compare:(NSString *)s options:(NSUInteger)opts;
- (BOOL)hasPrefix:(NSString *)prefix;
- (BOOL)hasSuffix:(NSString *)suffix;
- (BOOL)isEqualToString:(NSString *)s;
- (NSUInteger)lengthOfBytesUsingEncoding:(NSUInteger)enc;
- (NSArray *)componentsSeparatedByString:(NSString *)sep;
- (NSData *)dataUsingEncoding:(NSUInteger)enc;
- (instancetype)initWithData:(NSData *)data encoding:(NSUInteger)enc;
- (NSRange)rangeOfString:(NSString *)s;
- (NSRange)rangeOfString:(NSString *)s options:(NSUInteger)opts;
- (NSRange)rangeOfString:(NSString *)s options:(NSUInteger)opts range:(NSRange)r;
@property (readonly, copy) NSString *precomposedStringWithCanonicalMapping;
@property (readonly, copy) NSString *decomposedStringWithCanonicalMapping;
@end

@interface NSMutableString : NSString
+ (instancetype)string;
- (void)appendString:(NSString *)s;
- (void)appendFormat:(NSString *)fmt, ...;
- (void)setString:(NSString *)s;
@end

@interface NSArray<__covariant ObjectType> : NSObject <NSFastEnumeration>
+ (instancetype)arrayWithObjects:(const id[])objs count:(NSUInteger)cnt;
- (NSUInteger)count;
- (ObjectType)objectAtIndexedSubscript:(NSUInteger)idx;
- (ObjectType)objectAtIndex:(NSUInteger)idx;
- (ObjectType)firstObject;
- (ObjectType)lastObject;
- (NSArray *)sortedArrayUsingComparator:(NSComparator)cmp;
- (void)enumerateObjectsUsingBlock:(void (^)(id obj, NSUInteger idx, BOOL *stop))block;
@end

@interface NSMutableArray<ObjectType> : NSArray<ObjectType>
+ (instancetype)array;
+ (instancetype)arrayWithCapacity:(NSUInteger)cap;
- (void)addObject:(ObjectType)obj;
- (void)addObjectsFromArray:(NSArray *)other;
- (void)removeAllObjects;
- (void)removeLastObject;
- (void)insertObject:(ObjectType)obj atIndex:(NSUInteger)idx;
- (void)setObject:(ObjectType)obj atIndexedSubscript:(NSUInteger)idx;
- (void)replaceObjectAtIndex:(NSUInteger)idx withObject:(ObjectType)obj;
@end

@interface NSDictionary<__covariant KeyType, __covariant ObjectType> : NSObject <NSFastEnumeration>
+ (instancetype)dictionary;
+ (instancetype)dictionaryWithObjects:(const id[])objs
                              forKeys:(const id[])keys
                                count:(NSUInteger)cnt;
- (ObjectType)objectForKeyedSubscript:(KeyType)key;
- (ObjectType)objectForKey:(KeyType)key;
- (NSUInteger)count;
- (NSArray *)allKeys;
- (NSArray *)allValues;
- (void)enumerateKeysAndObjectsUsingBlock:(void (^)(KeyType key, ObjectType obj, BOOL *stop))block;
@end

@interface NSMutableDictionary<KeyType, ObjectType> : NSDictionary<KeyType, ObjectType>
- (void)setObject:(ObjectType)obj forKeyedSubscript:(KeyType)key;
- (void)removeObjectForKey:(KeyType)key;
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
- (BOOL)boolValue;
- (double)doubleValue;
- (NSInteger)integerValue;
@property (readonly) unsigned int unsignedIntValue;
@property (readonly) long long longLongValue;
@property (readonly) unsigned long long unsignedLongLongValue;
@end

@interface NSData : NSObject
+ (instancetype)dataWithBytes:(const void *)bytes length:(NSUInteger)len;
+ (instancetype)dataWithContentsOfFile:(NSString *)path;
- (const void *)bytes;
- (NSUInteger)length;
@end

@interface NSMutableData : NSData
+ (instancetype)data;
- (void)appendBytes:(const void *)bytes length:(NSUInteger)len;
- (void *)mutableBytes;
- (void)setLength:(NSUInteger)len;
@end

@interface NSJSONSerialization : NSObject
+ (id)JSONObjectWithData:(NSData *)data
                 options:(NSUInteger)opts
                   error:(NSError **)err;
@end

@interface NSValue : NSObject
+ (NSValue *)valueWithRange:(NSRange)range;
- (NSRange)rangeValue;
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
