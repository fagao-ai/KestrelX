package kestrel

/*
#include <stdlib.h>
#include "kestrel.h"
*/
import "C"

import (
	"encoding/json"
	"unsafe"
)

// ContractVersion returns the shared configuration schema version.
func ContractVersion() uint32 {
	return uint32(C.kestrel_contract_version())
}

// Capabilities returns capabilities supported by the linked FFI build.
func Capabilities() ([]string, error) {
	var errorC *C.char
	value := C.kestrel_capabilities_json(&errorC)
	if value == nil {
		return nil, takeCError(errorC)
	}
	defer C.kestrel_string_free(value)
	var capabilities []string
	if err := json.Unmarshal([]byte(C.GoString(value)), &capabilities); err != nil {
		return nil, err
	}
	return capabilities, nil
}

// ValidateConfig validates JSON against the shared schema and FFI capabilities.
func ValidateConfig(config any, hasCustomModel bool) error {
	data, err := encodeConfig(config)
	if err != nil {
		return err
	}
	configC := C.CString(string(data))
	defer C.free(unsafe.Pointer(configC))
	custom := C.int32_t(0)
	if hasCustomModel {
		custom = 1
	}
	var errorC *C.char
	if status := C.kestrel_validate_config(configC, custom, &errorC); status != 0 {
		return takeCError(errorC)
	}
	return nil
}
