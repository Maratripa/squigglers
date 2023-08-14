use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

#[proc_macro_derive(DistributionDerive)]
pub fn derive_distribution(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    // Ensure the input is an enum
    let data_enum = if let syn::Data::Enum(data_enum) = input.data {
        data_enum
    } else {
        panic!("Only enums are supported for deriving with DistributionDerive");
    };

    let enum_name = &input.ident;

    let variants = data_enum.variants.iter().map(|variant| {
        let variant_name = &variant.ident;

        // let fields = &variant.fields;
        quote! {
            #enum_name::#variant_name { dist } => dist.sample(rng)
        }
    });

    let gen = quote! {
        impl rand_distr::Distribution<f64> for #enum_name {
            fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> f64 {
                match self {
                    #(#variants),*
                }
            }

        }
    };

    gen.into()
}
