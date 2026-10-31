/* Smoke test for pe_ffi C ABI. Built and run from CI. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pe_ffi.h"

static int failures = 0;

#define EXPECT(cond)                                                         \
  do {                                                                       \
    if (!(cond)) {                                                           \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);        \
      failures++;                                                            \
    }                                                                        \
  } while (0)

int main(void) {
  const char *ver = pe_version_string();
  EXPECT(ver != NULL);
  EXPECT(strcmp(ver, "0.1.0") == 0);

  PeWorld *w = pe_world_create();
  EXPECT(w != NULL);
  EXPECT(pe_world_body_count(w) == 0);

  uint32_t idx = 999;
  int rc = pe_world_add_body(w, 1.0, 0.0, 0.0, 2.0, /*pos*/
                             0.0, 0.0, 0.0,          /*vel*/
                             1.0, 0.0, 0.0, 0.0,     /*quat*/
                             0.0, 0.0, 0.0,          /*omega*/
                             0.4, 0.4, 0.4,          /*I diag*/
                             0, 0.25, 0.0, 0.0,      /*sphere r=0.25*/
                             0.5, 0.3, &idx);
  EXPECT(rc == PE_OK);
  EXPECT(idx == 0);
  EXPECT(pe_world_body_count(w) == 1);

  uint32_t sidx = 999;
  rc = pe_world_add_static(w, 3, 0.0, 0.0, 1.0, 0.0, /*plane z=0*/
                           0.0, 0.0, 0.0, 0.0, 1.0, &sidx);
  EXPECT(rc == PE_OK);

  for (int i = 0; i < 120; i++) {
    rc = pe_world_step(w, 1.0 / 60.0);
    EXPECT(rc == PE_OK);
  }

  double pos[3];
  double quat[4];
  rc = pe_world_get_transforms(w, pos, quat, 1);
  EXPECT(rc == PE_OK);
  EXPECT(pos[2] < 2.0); /* fell under gravity */
  EXPECT(pos[2] > -0.5);

  int hit = 0;
  double t = -1.0;
  double point[3];
  double normal[3];
  uint32_t body = 99;
  rc = pe_world_raycast(w, 0.0, 0.0, 10.0, 0.0, 0.0, -1.0, 100.0, &hit, &t,
                        point, normal, &body);
  EXPECT(rc == PE_OK);
  EXPECT(hit == 1);
  EXPECT(t > 0.0);

  PeContactEvent contacts[32];
  size_t ncontacts = 0;
  rc = pe_world_get_contacts(w, contacts, 32, &ncontacts);
  EXPECT(rc == PE_OK);

  /* null destroy is fine */
  pe_world_destroy(NULL);
  pe_world_destroy(w);

  if (failures) {
    fprintf(stderr, "%d failure(s)\n", failures);
    return 1;
  }
  printf("pe_ffi C smoke OK\n");
  return 0;
}
