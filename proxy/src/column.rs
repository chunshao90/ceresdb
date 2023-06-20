// Copyright 2023 CeresDB Project Authors. Licensed under Apache-2.0.

//! A [`Column`] stores the rows for a given column name

use std::{fmt::Formatter, iter::Map, mem, path::Iter, sync::Arc, vec::IntoIter};

use arrow::{
    array::{
        ArrayDataBuilder, ArrayRef, BinaryArray, BooleanArray, Float64Array, Int64Array,
        StringArray, TimestampMillisecondArray, TimestampNanosecondArray, UInt64Array,
    },
    buffer::NullBuffer,
    datatypes::{DataType, TimeUnit},
    error::ArrowError,
};
use common_types::{
    datum::{Datum, DatumKind},
    string::StringBytes,
};
use snafu::{ResultExt, Snafu};

use crate::{bitset::BitSet, string::PackedStringArray};

/// A "dictionary ID" (DID) is a compact numeric representation of an interned
/// string in the dictionary. The same string always maps the same DID.
///
/// DIDs can be compared, hashed and cheaply copied around, just like small
/// integers.
///
/// An i32 is used to match the default for Arrow dictionaries
#[allow(clippy::upper_case_acronyms)]
pub(crate) type DID = i32;

/// An invalid DID used for NULL rows
pub(crate) const INVALID_DID: DID = -1;

#[derive(Debug, Snafu)]
#[allow(missing_copy_implementations, missing_docs)]
pub enum Error {
    #[snafu(display(
        "Invalid null mask, expected to be {} bytes but was {}",
        expected_bytes,
        actual_bytes
    ))]
    InvalidNullMask {
        expected_bytes: usize,
        actual_bytes: usize,
    },

    #[snafu(display("Internal MUB error constructing Arrow Array: {}", source))]
    CreatingArrowArray { source: ArrowError },
}

/// A specialized `Error` for [`Column`] errors
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Stores the actual data for columns in a chunk along with summary
/// statistics
#[derive(Debug, Clone)]
pub struct Column {
    pub(crate) datum_kind: DatumKind,
    pub(crate) valid: BitSet,
    pub(crate) data: ColumnData,
    pub(crate) to_insert: usize,
}

/// The data for a column
#[derive(Debug, Clone)]
#[allow(missing_docs)]
pub enum ColumnData {
    F64(Vec<f64>),
    I64(Vec<i64>),
    U64(Vec<u64>),
    String(Vec<StringBytes>),
    StringRaw(Vec<String>),
    Bool(BitSet),
}

// impl Iterator for ColumnData{
//     type Item = Datum;
//
//     fn next(&mut self) -> Option<Self::Item> {
//         match self {
//             Self::F64(col_data) => col_data.pop().map(|x| Datum::Double(x)),
//             Self::I64(col_data) => col_data.pop().map(|x| Datum::Int64(x)),
//             Self::U64(col_data) => col_data.pop().map(|x| Datum::UInt64(x)),
//             Self::String(col_data) => col_data
//                 .pop()
//                 .map(|x| Datum::String(StringBytes::from(x))),
//             Self::Bool(col_data) => todo!(),
//         }
//     }
// }

pub enum ColumnDataIter {
    F64(std::vec::IntoIter<f64>),
    I64(std::vec::IntoIter<i64>),
    U64(std::vec::IntoIter<u64>),
    String(std::vec::IntoIter<StringBytes>),
    StringRaw(std::vec::IntoIter<String>),
}

impl<'a> Iterator for ColumnDataIter {
    type Item = Datum;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::F64(col_data) => col_data.next().map(|x| Datum::Double(x)),
            Self::I64(col_data) => col_data.next().map(|x| Datum::Int64(x)),
            Self::U64(col_data) => col_data.next().map(|x| Datum::UInt64(x)),
            Self::String(col_data) => col_data.next().map(|x| Datum::String(x.into())),
            Self::StringRaw(col_data) => col_data.next().map(|x| Datum::StringRaw(x)),
            // Self::Bool(col_data) => todo!(),
        }
    }
}

impl IntoIterator for ColumnData {
    type IntoIter = ColumnDataIter;
    type Item = Datum;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            Self::F64(col_data) => ColumnDataIter::F64(col_data.into_iter()),
            Self::I64(col_data) => ColumnDataIter::I64(col_data.into_iter()),
            Self::U64(col_data) => ColumnDataIter::U64(col_data.into_iter()),
            Self::String(col_data) => ColumnDataIter::String(col_data.into_iter()),
            Self::StringRaw(col_data) => ColumnDataIter::StringRaw(col_data.into_iter()),
            Self::Bool(col_data) => todo!(),
        }
    }
}

impl ColumnData {
    // pub fn pop(&mut self) -> Datum {
    //     match self {
    //         Self::F64(col_data) => Datum::Double(col_data.remove(col_data.len() -
    // 1)),         Self::I64(col_data) =>
    // Datum::Int64(col_data.remove(col_data.len() - 1)),
    //         Self::U64(col_data) => Datum::UInt64(col_data.remove(col_data.len() -
    // 1)),         Self::String(col_data) => {
    //             Datum::String(StringBytes::from(col_data.remove(col_data.len() -
    // 1)))         }
    //         Self::Bool(col_data) => Datum::Boolean(col_data.get(0)),
    //     }
    // }

    // pub fn get(&self, idx: usize) -> Datum {
    //     match self {
    //         Self::F64(col_data) => Datum::Double(col_data[idx]),
    //         Self::I64(col_data) => Datum::Int64(col_data[idx]),
    //         Self::U64(col_data) => Datum::UInt64(col_data[idx]),
    //         Self::String(col_data) => {
    //
    // Datum::String(StringBytes::copy_from_str(col_data.get(idx).unwrap()))
    //         }
    //         Self::Bool(col_data) => Datum::Boolean(col_data.get(idx)),
    //     }
    // }

    // pub fn append(&mut self, datum: Datum) {
    //     match (self, datum) {
    //         (Self::I64(col_data), Datum::Timestamp(datum)) =>
    // col_data.push(datum.as_i64()),         (Self::F64(col_data),
    // Datum::Double(datum)) => col_data.push(datum),
    //         (Self::I64(col_data), Datum::Int64(datum)) => col_data.push(datum),
    //         (Self::U64(col_data), Datum::UInt64(datum)) => col_data.push(datum),
    //         (Self::String(col_data), Datum::String(datum)) =>
    // col_data.push(datum.to_string()),         (Self::Bool(col_data),
    // Datum::Boolean(datum)) => {            todo!()
    //         }
    //         _ => panic!("Invalid datum type"),
    //     }
    // }
}

impl std::fmt::Display for ColumnData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::F64(col_data) => write!(f, "F64({})", col_data.len()),
            Self::I64(col_data) => write!(f, "I64({})", col_data.len()),
            Self::U64(col_data) => write!(f, "U64({})", col_data.len()),
            Self::String(col_data) => write!(f, "String({})", col_data.len()),
            Self::Bool(col_data) => write!(f, "Bool({})", col_data.len()),
            _ => todo!(),
        }
    }
}

impl Column {
    pub(crate) fn new(row_count: usize, kind: DatumKind) -> Self {
        let mut valid = BitSet::new();
        valid.append_unset(row_count);

        // Keep track of how many total rows there are
        let total_count = row_count as u64;

        let data = match kind {
            DatumKind::Boolean => {
                let mut data = BitSet::new();
                data.append_unset(row_count);
                ColumnData::Bool(data)
            }
            DatumKind::UInt64 => ColumnData::U64(vec![0; row_count]),
            DatumKind::Float => ColumnData::F64(vec![0.0; row_count]),
            DatumKind::Int64 | DatumKind::Timestamp => ColumnData::I64(vec![0; row_count]),
            DatumKind::String => ColumnData::String(vec![StringBytes::new(); row_count]),
            _ => todo!(),
        };

        Self {
            datum_kind: kind,
            valid,
            data,
            to_insert: 0,
        }
    }

    pub fn append(&mut self, datum: Datum) {
        match self.data {
            ColumnData::I64(ref mut data) => match datum {
                Datum::Int64(datum) => data[self.to_insert] = datum,
                Datum::Timestamp(datum) => data[self.to_insert] = datum.as_i64(),
                _ => panic!("Invalid datum type"),
            },
            ColumnData::F64(ref mut data) => {
                data[self.to_insert] = datum.as_f64().unwrap();
            }
            ColumnData::U64(ref mut data) => {
                data[self.to_insert] = datum.as_u64().unwrap();
            }
            ColumnData::String(ref mut data) => {
                if let Datum::String(v) = datum {
                    data[self.to_insert] = v;
                }
                // datum.do_with_bytes(|bytes| {
                //     data[self.to_insert] =
                // String::from_utf8_lossy(bytes).to_string();
                // });
                // data[self.to_insert] = datum.to_bytes().unwrap().to_string();
            }
            ColumnData::StringRaw(ref mut data) => {
                if let Datum::StringRaw(v) = datum {
                    data[self.to_insert] = v;
                }
            }
            ColumnData::Bool(ref mut data) => {
                todo!()
            }
        }
        self.valid.set(self.to_insert);
        self.to_insert += 1;
    }

    // pub fn append(&mut self, datum: Datum) {
    //     self.data.append(datum);
    //     self.valid.append_set(1);
    // }

    /// Returns the validity bitmask of this column
    pub fn valid_mask(&self) -> &BitSet {
        &self.valid
    }

    /// Returns a reference to this column's data
    pub fn data(&self) -> &ColumnData {
        &self.data
    }

    /// Ensures that the total length of this column is `len` rows,
    /// padding it with trailing NULLs if necessary
    pub(crate) fn push_nulls_to_len(&mut self, len: usize) {
        if self.valid.len() == len {
            return;
        }
        assert!(len > self.valid.len(), "cannot shrink column");
        let delta = len - self.valid.len();
        self.valid.append_unset(delta);

        match &mut self.data {
            ColumnData::F64(data) => {
                data.resize(len, 0.);
            }
            ColumnData::I64(data) => {
                data.resize(len, 0);
            }
            ColumnData::U64(data) => {
                data.resize(len, 0);
            }
            ColumnData::String(data) => {
                // data.resize(len, "".to_string());
                todo!()
            }
            ColumnData::Bool(data) => {
                data.append_unset(delta);
            }
            _ => todo!(),
        }
    }

    /// Returns the number of rows in this column
    pub fn len(&self) -> usize {
        self.valid.len()
    }

    /// Returns true if this column contains no rows
    pub fn is_empty(&self) -> bool {
        self.valid.is_empty()
    }

    /// The approximate memory size of the data in the column.
    ///
    /// This includes the size of `self`.
    pub fn size(&self) -> usize {
        let data_size = match &self.data {
            ColumnData::F64(v) => mem::size_of::<f64>() * v.capacity(),
            ColumnData::I64(v) => mem::size_of::<i64>() * v.capacity(),
            ColumnData::U64(v) => mem::size_of::<u64>() * v.capacity(),
            ColumnData::Bool(v) => v.byte_len(),
            ColumnData::String(v) => todo!(),
            _ => todo!(),
        };
        mem::size_of::<Self>() + data_size + self.valid.byte_len()
    }

    /// The approximate memory size of the data in the column, not counting for
    /// stats or self or whatever extra space has been allocated for the
    /// vecs
    pub fn size_data(&self) -> usize {
        match &self.data {
            ColumnData::F64(_) => mem::size_of::<f64>() * self.len(),
            ColumnData::I64(_) => mem::size_of::<i64>() * self.len(),
            ColumnData::U64(_) => mem::size_of::<u64>() * self.len(),
            ColumnData::Bool(_) => mem::size_of::<bool>() * self.len(),
            ColumnData::String(v) => todo!(),
            _ => todo!(),
        }
    }

    /// Converts this column to an arrow [`ArrayRef`]
    pub fn to_arrow(&self) -> Result<ArrayRef> {
        let nulls = Some(NullBuffer::new(self.valid.to_arrow()));

        let data: ArrayRef = match &self.data {
            ColumnData::F64(data) => {
                let data = ArrayDataBuilder::new(DataType::Float64)
                    .len(data.len())
                    .add_buffer(data.iter().cloned().collect())
                    .nulls(nulls)
                    .build()
                    .unwrap();
                Arc::new(Float64Array::from(data))
            }
            ColumnData::I64(data) => match self.datum_kind {
                DatumKind::Timestamp => {
                    let data =
                        ArrayDataBuilder::new(DataType::Timestamp(TimeUnit::Millisecond, None))
                            .len(data.len())
                            .add_buffer(data.iter().cloned().collect())
                            .nulls(nulls)
                            .build()
                            .unwrap();
                    Arc::new(TimestampMillisecondArray::from(data))
                }

                DatumKind::Int64 => {
                    let data = ArrayDataBuilder::new(DataType::Int64)
                        .len(data.len())
                        .add_buffer(data.iter().cloned().collect())
                        .nulls(nulls)
                        .build()
                        .unwrap();
                    Arc::new(Int64Array::from(data))
                }
                _ => unreachable!(),
            },
            ColumnData::U64(data) => {
                let data = ArrayDataBuilder::new(DataType::UInt64)
                    .len(data.len())
                    .add_buffer(data.iter().cloned().collect())
                    .nulls(nulls)
                    .build()
                    .unwrap();
                Arc::new(UInt64Array::from(data))
            }
            ColumnData::String(data) => {
                let data =
                    StringArray::from(data.iter().map(|s| Some(s.as_str())).collect::<Vec<_>>());
                Arc::new(data)
            }
            ColumnData::Bool(data) => {
                let data = ArrayDataBuilder::new(DataType::Boolean)
                    .len(data.len())
                    .add_buffer(data.to_arrow().into_inner())
                    .nulls(nulls)
                    .build()
                    .unwrap();
                Arc::new(BooleanArray::from(data))
            }
            _ => todo!(),
        };
        // println!("to_arrow: {:?}", data);

        assert_eq!(data.len(), self.len());

        Ok(data)
    }
}
